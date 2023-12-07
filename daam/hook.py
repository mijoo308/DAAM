import functools
import itertools

from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.attention_processor import Attention


class ObjectHooker:
    def __init__(self, module: list) -> None:
        self.module = module
        self.hooked = False
        self.old_state = dict()
        
    def __enter__(self):
        # When this class called at 'with', this function called.
        self.hook()
        return self
    
    def __exit__(self, type, value, trackback):
        # When this class is closed at 'with', this function called.
        # [mj] argument should exist even if not used explicitly.
        self.unhook()
    
    def hook(self):
        if self.hooked:
            raise RuntimeError('This module is already hooked.')
        
        self.old_state = dict()
        self.hooked = True
        self._hook_impl()
        return self
    
    def unhook(self):
        if not self.hooked:
            raise RuntimeError('This module is not hooked.')

        for key, value in self.old_state.items():
            if key.startswith('old_fn_'):
                setattr(self.module, key[7:], value)
        self.hooked = False
        self._unhook_impl()
        return self
    
    def _hook_impl(self):
        pass
    
    def _unhook_impl(self):
        pass

    def monkey_patch(self, fn_name, fn):
        self.old_state[f'old_fn_{fn_name}'] = getattr(self.module, fn_name)
        setattr(self.module, fn_name, functools.partial(fn, args=self.module))

    def monkey_super(self, fn_name, *args, **kwargs):
        return self.old_state[f'old_fn_{fn_name}'](*args, **kwargs)
    

class ModuleLocator:
    def __init__(self) -> None:
        pass
    
    def locate(self, model):
        raise NotImplementedError


class AggregateHooker(ObjectHooker):
    def _hook_impl(self):
        for h in self.module:
            h.hook()
    
    def _unhook_impl(self):
        for h in self.module:
            h.unhook()
    
    def register_hook(self, hook):
        self.module.append(hook)


class UNetCrossAttentionLocator(ModuleLocator):
    def __init__(self, restrict=None, locate_middle_block=False) -> None:
        self.restrict = restrict
        self.layer_names = list()
        self.locate_middle_block = locate_middle_block
        
    def locate(self, model: UNet2DConditionModel) -> list:
        """
        Locate all cross-attention modules in a UNet2DConditionModel.

        Args:
            model (`UNet2DConditionModel`): The model to locate the cross-attention modules in.

        Returns:
            `List[Attention]`: The list of cross-attention modules.
        """
        self.layer_names.clear()
        blocks_list = list()
        up_names = ['up'] * len(model.up_blocks)
        down_names = ['down'] * len(model.down_blocks)
        
        for block, name in zip(model.up_blocks, up_names):
            if 'CrossAttn' in block.__class__.__name__:
                blocks_list.extend(self.cross_attn(block, name))

        for block, name in zip(model.down_blocks, down_names):
            if 'CrossAttn' in block.__class__.__name__:
                blocks_list.extend(self.cross_attn(block, name))

        for block, name in zip([model.mid_block], ['mid']) if self.locate_middle_block else []:
            if 'CrossAttn' in block.__class__.__name__:
                blocks_list.extend(self.cross_attn(block, name))

        return blocks_list

    def cross_attn(self, block, name):
        blocks = []

        for spatial_transformer in block.attentions:
            for transformer_block in spatial_transformer.transformer_blocks:
                blocks.append(transformer_block.attn2)

        blocks = [b for idx, b in enumerate(blocks) if self.restrict is None or idx in self.restrict]
        names = [f'{name}-attn-{i}' for i in range(len(blocks)) if self.restrict is None or i in self.restrict]
        self.layer_names.extend(names)
        return blocks