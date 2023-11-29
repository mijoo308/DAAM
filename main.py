import sys
import argparse
import os

import torch
from daam import set_seed, trace, hook
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[NLP 23f project] DAAM Implementation")
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-base', help='model id')
    parser.add_argument('--prompt', '-p', type=str, default='Two dogs run across the field', help='prompt (sentence)')
    parser.add_argument('--word', '-w', type=str, default='Two', help='word to be attentioned')
    parser.add_argument('--num_inference', '-it', type=int, default=30, help='set the number for inference step')
    parser.add_argument('--save', '-s', type=str, default='./result', help='save path')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print('This code may unavailable with cpu.')


    if not os.path.exists(args.save):
        os.mkdir(args.save)
    

    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, use_auth_token=True)
    pipe = pipe.to(device)

    save_path = f'{args.save}/{args.prompt} ({args.word}).png'
    seed = set_seed(0)
    with torch.no_grad():
        with trace.DiffusionHeatMapHooker(pipe) as HOOKER:
            out = pipe(args.prompt, num_inference_steps=args.num_inference, generator=seed) # out.images[0].shape : 64x64

            # core 
            heat_map = HOOKER.compute_global_heat_map()
            heat_map = heat_map.compute_word_heat_map(args.word)

            # visualization
            heat_map.plot_overlay(out.images[0])
            plt.show()
            plt.savefig(save_path)
