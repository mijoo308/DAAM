import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from matplotlib import pyplot as plt
from typing import List, Any, Dict, Tuple, Set, Iterable, Union, Optional
import spacy.tokens
from spacy.tokens import Token

from .evaluate import compute_ioa
from .utils import compute_token_merge_indices, cached_nlp, auto_autocast

__all__ = ['GlobalHeatMap', 'RawHeatMapCollection', 'WordHeatMap', 'ParsedHeatMap', 'SyntacticHeatMapPair']


def plot_overlay_heat_map(image: Union[PIL.Image.Image, np.ndarray], 
                          heat_map: torch.Tensor, 
                          word: Optional[str] = None, 
                          out_file: Optional[Union[str, Path]] = None, 
                          crop: Optional[int] = None, 
                          color_normalize: bool = True, 
                          ax: Optional[plt.Axes] = None):
    plt_ = setup_plot(ax)
    image_processed, heat_map_processed = process_image_heatmap(image, heat_map, crop, color_normalize)
    plot_heatmap_image_overlay(image_processed, heat_map_processed, plt_)
    set_title_and_save_plot(word, out_file, plt_, ax)

def setup_plot(ax: Optional[plt.Axes]) -> plt.Axes:
    if ax is None:
        plt.clf()
        plt.rcParams.update({'font.size': 24})
        return plt
    return ax

def process_image_heatmap(image: Union[PIL.Image.Image, np.ndarray], 
                          heat_map: torch.Tensor, 
                          crop: Optional[int], 
                          color_normalize: bool) -> (np.ndarray, torch.Tensor):
    image_np = np.array(image)
    heat_map_np = heat_map.cpu().numpy() if color_normalize else heat_map.clamp_(min=0, max=1).cpu().numpy()
    if crop is not None:
        image_np = image_np[crop:-crop, crop:-crop]
        heat_map_np = heat_map_np[crop:-crop, crop:-crop]
    return image_np, torch.tensor(heat_map_np)

def plot_heatmap_image_overlay(image_np: np.ndarray, heat_map_tensor: torch.Tensor, plt_: plt.Axes):
    plt_.imshow(heat_map_tensor, cmap='jet')
    overlay = create_overlay(image_np, heat_map_tensor)
    plt_.imshow(overlay)

def create_overlay(image_np: np.ndarray, heat_map_tensor: torch.Tensor) -> np.ndarray:
    image_tensor = torch.from_numpy(image_np).float() / 255
    return torch.cat((image_tensor, (1 - heat_map_tensor.unsqueeze(-1))), dim=-1).numpy()

def set_title_and_save_plot(word: Optional[str], out_file: Optional[Union[str, Path]], plt_: plt.Axes, ax: Optional[plt.Axes]):
    if word:
        title_function = plt_.title if ax is None else ax.set_title
        title_function(word)
    if out_file:
        plt_.savefig(out_file)




class WordHeatMap:
    def __init__(self, heatmap: torch.Tensor, word: Optional[str] = None, word_idx: Optional[int] = None):
        self.word = word
        self.word_idx = word_idx
        self.heatmap = heatmap

    @property
    def value(self) -> torch.Tensor:
        return self.heatmap

    def plot_overlay(self, image: Union[PIL.Image.Image, Any], out_file: Optional[str] = None,
                     color_normalize: bool = True, ax: Optional[Any] = None, **expand_kwargs):
        expanded_heatmap = self.expand_as(image, **expand_kwargs)
        plot_overlay_heat_map(image, expanded_heatmap, word=self.word, out_file=out_file,
                              color_normalize=color_normalize, ax=ax)

    def expand_as(self, image: Union[PIL.Image.Image, Any], absolute: bool = False,
                  threshold: Optional[float] = None) -> torch.Tensor:
        interpolated_heatmap = self.interpolate_heatmap(image)
        normalized_heatmap = self.normalize_heatmap(interpolated_heatmap, absolute)
        return self.apply_threshold(normalized_heatmap, threshold) if threshold is not None else normalized_heatmap

    def interpolate_heatmap(self, image: Union[PIL.Image.Image, Any]) -> torch.Tensor:
        image_size = self.get_image_size(image)

        # Check if the heatmap is 1D or 2D and reshape it accordingly
        if len(self.heatmap.shape) == 1:
            # Reshape 1D heatmap to 2D (assuming it's a square)
            side_length = int(math.sqrt(len(self.heatmap)))
            heatmap_reshaped = self.heatmap.view(1, 1, side_length, side_length)
        elif len(self.heatmap.shape) == 2:
            # Add a channel dimension to 2D heatmap
            heatmap_reshaped = self.heatmap.unsqueeze(0).unsqueeze(0).float()
        elif len(self.heatmap.shape) == 3:
            heatmap_reshaped = self.heatmap.unsqueeze(0).float()
        else:
            heatmap_reshaped = self.heatmap.float()

        return F.interpolate(heatmap_reshaped, size=image_size, mode='bicubic').cpu().squeeze()

    def normalize_heatmap(self, heatmap: torch.Tensor, absolute: bool) -> torch.Tensor:
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) if not absolute else heatmap

    def apply_threshold(self, heatmap: torch.Tensor, threshold: float) -> torch.Tensor:
        return (heatmap > threshold).float()

    def get_image_size(self, image: Union[PIL.Image.Image, Any]) -> tuple:
        return (image.size[0], image.size[1]) if isinstance(image, PIL.Image.Image) else image.shape[:2]

    def compute_ioa(self, other: 'WordHeatMap') -> float:
        return compute_ioa(self.heatmap, other.heatmap)



@dataclass
class SyntacticHeatMapPair:
    head_heat_map: WordHeatMap
    dep_heat_map: WordHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class ParsedHeatMap:
    word_heat_map: WordHeatMap
    token: spacy.tokens.Token


class GlobalHeatMap:
    def __init__(self, tokenizer: Any, prompt: str, heat_maps: torch.Tensor):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.heat_maps = heat_maps
        self._compute_word_heat_map = lru_cache(maxsize=50)(self._compute_word_heat_map_internal)

    def compute_word_heat_map(self, word: str, word_idx: int = None) -> WordHeatMap:
        return self._compute_word_heat_map(word, word_idx)

    def _compute_word_heat_map_internal(self, word: str, word_idx: int = None) -> WordHeatMap:
        merge_idxs = compute_token_merge_indices(self.tokenizer, self.prompt, word, word_idx)
        averaged_heatmap = self.heat_maps[merge_idxs].mean(0)
        return WordHeatMap(averaged_heatmap, word, word_idx)

    def generate_parsed_heat_maps(self) -> Iterable[ParsedHeatMap]:
        for token in cached_nlp(self.prompt):
            heat_map = self.compute_word_heat_map(token.text)
            if heat_map:
                yield ParsedHeatMap(heat_map, token)

    def generate_syntactic_relations(self) -> Iterable[SyntacticHeatMapPair]:
        for token in cached_nlp(self.prompt):
            if token.dep_ != 'ROOT':
                dep_heat_map = self.compute_word_heat_map(token.text)
                head_heat_map = self.compute_word_heat_map(token.head.text)
                if dep_heat_map and head_heat_map:
                    yield SyntacticHeatMapPair(head_heat_map, dep_heat_map, token.head.text, token.text, token.dep_)


RawHeatMapKey = Tuple[int, int, int]  # factor, layer, head


class RawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)

    def update(self, factor: int, layer_idx: int, head_idx: int, heatmap: torch.Tensor):
        with auto_autocast(dtype=torch.float32):
            key = (factor, layer_idx, head_idx)
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
