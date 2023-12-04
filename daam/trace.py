from pathlib import Path
from typing import List, Type, Any, Dict, Tuple, Union
import math

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F

from .utils import cache_dir, auto_autocast
from .experiment import GenerationExperiment
from .heatmap import RawHeatMapCollection, GlobalHeatMap
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator

LATENT_SIZE_512 = 4096
LATENT_SIZE_OTHER = 9216

class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(self, pipeline, low_memory=False, load_heads=False, save_heads=False, data_dir=None):
        self.all_heat_maps = RawHeatMapCollection()
        h = (pipeline.unet.config.sample_size * pipeline.vae_scale_factor)
        latent_size = LATENT_SIZE_512 if h == 512 else LATENT_SIZE_OTHER
        self.latent_hw = latent_size
        locate_middle = load_heads or save_heads #?
        self.locator = UNetCrossAttentionLocator(restrict=None, locate_middle_block=locate_middle) # Q.
        self.last_prompt = ''
        self.last_image = None
        self.time_idx = 0
        self._gen_idx = 0


        modules = self._generate_modules(pipeline, load_heads, save_heads, data_dir)
        super().__init__(modules)
        self.pipe = pipeline

    def _generate_modules(self, pipeline, load_heads, save_heads, data_dir):
        modules = [
            UNetCrossAttentionHooker(
                x, self, layer_idx=idx, latent_hw=self.latent_hw, load_heads=load_heads, save_heads=save_heads, data_dir=data_dir
            ) for idx, x in enumerate(self.locator.locate(pipeline.unet))
        ]
        modules.append(PipelineHooker(pipeline, self))
        return modules


    def compute_global_heat_map(self, prompt=None, factors=None, head_idx=None,
                                layer_idx=None, normalize=False):
        heat_maps = self.all_heat_maps
        prompt = prompt or self.last_prompt
        factors = set(factors) if factors is not None else {0, 1, 2, 4, 8, 16, 32, 64}

        all_merges = self._generate_merged_heat_maps(heat_maps, factors, head_idx, layer_idx)

        maps = torch.stack(all_merges, dim=0).mean(0)[:, 0]
        maps = self._adjust_maps_length(maps, prompt)

        if normalize:
            maps = self._normalize_maps(maps)

        return GlobalHeatMap(self.pipe.tokenizer, prompt, maps)


    def _generate_merged_heat_maps(self, heat_maps, factors, head_idx, layer_idx):
        x = int(np.sqrt(self.latent_hw))
        all_merges = []
        for (factor, layer, head), heat_map in heat_maps:
            if self._is_relevant_heat_map(factor, head, layer, factors, head_idx, layer_idx):
                all_merges.append(self._interpolate_heat_map(heat_map, x))
        return all_merges

    def _is_relevant_heat_map(self, factor, head, layer, factors, head_idx, layer_idx):
        return (factor in factors and
                (head_idx is None or head_idx == head) and
                (layer_idx is None or layer_idx == layer))

    def _interpolate_heat_map(self, heat_map, size):
        heat_map = heat_map.unsqueeze(1)  # Adding an extra dimension
        return F.interpolate(heat_map, size=(size, size), mode='bicubic').clamp_(min=0)

    def _adjust_maps_length(self, maps, prompt):
        tokenized_length = len(self.pipe.tokenizer.tokenize(prompt)) + 2  # Account for SOS and padding
        return maps[:tokenized_length]

    def _normalize_maps(self, maps):
        return maps / (maps[1:-1].sum(0, keepdim=True) + 1e-6)


class PipelineHooker(ObjectHooker):
    def __init__(self, pipeline : StableDiffusionPipeline, parent_trace):
        super().__init__(pipeline)
        self.heat_maps = parent_trace.all_heat_maps
        self.parent_trace = parent_trace

    def _hooked_run_safety_checker(hk_self, image, pipe, *args, **kwargs):
        # [mj] arguments come with wierd order ..  
        # so manually fixed it
        args = ((pipe,) + args[:])
        pipe = kwargs.pop('args')

        image, has_nsfw = hk_self.monkey_super('run_safety_checker', image, *args, **kwargs)

        pil_image = pipe.numpy_to_pil(image)
        hk_self.parent_trace.last_image = pil_image[0]

        return image, has_nsfw

    def _hooked_encode_prompt(hk_self, prompt, pipe, *args, **kwargs):
        # [mj] arguments come with wierd order .. 
        # so manually fixed it
        pipeline = kwargs.pop('args')
        args = ((pipe,)+ args[:])
        # prompt = _

        if not isinstance(prompt, str) and len(prompt) > 1:
            raise ValueError('Only single prompt generation is supported for heat map computation.')
        elif not isinstance(prompt, str):
            last_prompt = prompt[0]
        else:
            last_prompt = prompt

        hk_self.heat_maps.clear()
        hk_self.parent_trace.last_prompt = last_prompt
        ret = hk_self.monkey_super('_encode_prompt', prompt, *args, **kwargs) # prompt : 'Two dogs run across the field'

        return ret

    def _hook_impl(self):
        self.monkey_patch('run_safety_checker', self._hooked_run_safety_checker)
        self.monkey_patch('_encode_prompt', self._hooked_encode_prompt)


class UNetCrossAttentionHooker(ObjectHooker):
    def __init__(self, module, parent_trace, context_size=77, layer_idx=0,
                 latent_hw=9216, load_heads=False, save_heads=False,
                 data_dir=None):
        super().__init__(module)
        self.heat_maps = parent_trace.all_heat_maps
        self.context_size = context_size
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw

        self.load_heads = load_heads
        self.save_heads = save_heads
        self.trace = parent_trace

        self.data_dir = Path(data_dir) if data_dir else cache_dir() / 'heads'
        self.data_dir.mkdir(parents=True, exist_ok=True)


    def _unravel_attn(self, attention_probs):
        with torch.no_grad():
            h = w = int(math.sqrt(attention_probs.size(1)))
            maps = []
            attention_probs = attention_probs.permute(2, 0, 1)

            for attention_map in attention_probs:
                attention_map = attention_map.view(attention_map.size(0), h, w)
                attention_map = attention_map[attention_map.size(0) // 2:]  # Filter out unconditional
                maps.append(attention_map)

            return torch.stack(maps, 0).permute(1, 0, 2, 3).contiguous()  # shape: (heads, tokens, height, width)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        """Process the attention mechanism and update heat maps."""
        batch_size, sequence_length, _ = hidden_states.shape # debug : torch.Size([2, 4096, 320])
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query, key, value = self._prepare_attention_components(attn, hidden_states, encoder_hidden_states)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self._handle_attention_saving(attention_probs)

        # compute shape factor
        factor = int(math.sqrt(self.latent_hw // attention_probs.shape[1]))
        self.trace._gen_idx += 1

        maps = self._unravel_attn(attention_probs)

        self._update_heat_maps(maps, factor)
        hidden_states = self._apply_attention_to_hidden_states(attn, attention_probs, value)

        return hidden_states

    def _prepare_attention_components(self, attn, hidden_states, encoder_hidden_states):
        """Prepare query, key, value components for attention calculation."""
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        return query, key, value

    def _handle_attention_saving(self, attention_probs):
        """Save or load attention probabilities if required."""
        if self.save_heads:
            self._save_attn(attention_probs)
        elif self.load_heads:
            return self._load_attn()
        
    def _update_heat_maps(self, maps, factor):
        """Update the heat maps collection with new data."""
        for head_idx, heatmap in enumerate(maps):
            self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)

    def _apply_attention_to_hidden_states(self, attn, attention_probs, value):
        """Apply attention probabilities to hidden states."""
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        for out_module in attn.to_out:
            hidden_states = out_module(hidden_states)
        return hidden_states

    def _hook_impl(self):
        self.module.set_processor(self)
