# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# Reference image = 1, inference from reference image
"""Generate images using pretrained network pickle."""
import cv2
import pyspng
import glob
import os
import re
import random
from typing import List, Optional
from tqdm import tqdm
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import legacy
from datasets.mask_generator_256 import RandomMask, OutpaintMask
from networks.mat import Generator


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dpath', help='the path of the input image', required=True)
@click.option('--mpath', help='the path of the mask')
@click.option('--resolution', type=int, help='resolution of input image', default=256, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    dpath: str,
    mpath: Optional[str],
    resolution: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
):
    """
    Generate images using pretrained network pickle.
    """
    seed = 202  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Loading data from: {dpath}')
    img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg'))

    if mpath is not None:
        print(f'Loading mask from: {mpath}')
        mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg'))
        assert len(img_list) == len(mask_list), 'illegal mapping'

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=False)

    os.makedirs(outdir, exist_ok=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        
        return image


    def mask_generator(mask):
    # base_mask = np.ones([256,256], np.uint8)
    
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, dst = cv2.threshold(gray, 1, 1, cv2.THRESH_BINARY)
        dst = dst.astype(dtype = 'uint8')
        return dst[np.newaxis, ...].astype(np.float32)

    if resolution != 512:
        noise_mode = 'random'

# CROSS SHAPE OUTPAINTING
    
    with torch.no_grad():
        for k, ipath in enumerate(img_list):
            iname = os.path.basename(ipath).replace('.jpg', '.png')
            print(f'Prcessing: {iname}')
            image = read_image(ipath)

            outpaint_res = resolution
            steps = 512

            palette_texture = np.zeros([outpaint_res,outpaint_res,3], dtype='uint8') # palette texture generation
            palette_texture[256:768,256:768,:] = image            
        
            for i in range(2):
                input = palette_texture[i*steps:(i+1)*steps,256:768,:]

                mask = mask_generator(input)
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

                input = input.transpose(2, 0, 1) # HWC => CHW
                input = input[:3]
                input = (torch.from_numpy(input).float().to(device) / 127.5 - 1).unsqueeze(0)

                z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
                output = G(input, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                output = output[0].cpu().numpy()
                palette_texture[i*steps:(i+1)*steps,256:768,:] = output
            
            for i in range(2):
                input = palette_texture[256:768, i*steps:(i+1)*steps,:]

                mask = mask_generator(input)
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

                input = input.transpose(2, 0, 1) # HWC => CHW
                input = input[:3]
                input = (torch.from_numpy(input).float().to(device) / 127.5 - 1).unsqueeze(0)

                z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
                output = G(input, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                output = output[0].cpu().numpy()
                palette_texture[256:768, i*steps:(i+1)*steps,:] = output

# EDGE INFO 2 CENTER INFO GENERATION
            palette_texture_2 = np.zeros([1024,1024,3], dtype='uint8')
            # palette_texture[palette_texture==0] = 255
            palette_texture_2[64:960,64:960,:] = palette_texture[64:960,64:960,:]
            palette_texture = np.roll(palette_texture,512,axis=(0,1)) 

            for i in range(4):
                palette_texture = np.rot90(palette_texture, i)
                endpoint = np.zeros([512,512,3], dtype='uint8')
                endpoint[0:384, 0:384, :] = palette_texture[128:512, 128:512, :]
                endpoint[128:384, 384:512, :] = palette_texture[256:512, 768:896, :]
                endpoint[384:512, 128:384, :] = palette_texture[768:896, 256:512, :]
                
                input = endpoint
                
                mask = mask_generator(input)
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

                input = input.transpose(2, 0, 1) # HWC => CHW
                input = input[:3]
                input = (torch.from_numpy(input).float().to(device) / 127.5 - 1).unsqueeze(0)

                z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
                output = G(input, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                output = output[0].cpu().numpy()

                palette_texture[256:512,256:512,:] = output[128:384,128:384,:]

# CROSS INPAINTING

            palette_texture = np.rot90(palette_texture, 1)
            palette_texture[512-96:512+96,:,:] = 0
#             PIL.Image.fromarray(palette_texture, 'RGB').save(f'{outdir}/{os.path.basename(ipath)}_4_first_line.png')

            
            for i in range(0,4):
                palette_texture_pad = np.pad(palette_texture, ((128,128),(128,128),(0, 0)), 'constant', constant_values=0)
                input = palette_texture_pad[640-256:640+256, i*256:i*256+512,:]
                if i == 3:
                    input[:,384:,:] = palette_texture_pad[640-256:640+256, 128:256, :]
#                 PIL.Image.fromarray(input, 'RGB').save(f'{outdir}/{os.path.basename(ipath)}_2_{i}_input.png')

                mask = mask_generator(input)
                if mask.min() == 1:
                    continue
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

                input = input.transpose(2, 0, 1) # HWC => CHW
                input = input[:3]
                input = (torch.from_numpy(input).float().to(device) / 127.5 - 1).unsqueeze(0)

                z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
                output = G(input, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                output = output[0].cpu().numpy()
                palette_texture[512-96:512+96, i*256:(i+1)*256,:] = output[256-96:256+96,256-128:256+128,:]


            palette_texture_2 = np.rot90(palette_texture, 1)
            palette_texture_2[512-96:512+96,:,:] = 0
            
            for i in range(0,4):
                palette_texture_pad_2 = np.pad(palette_texture_2, ((128,128),(128,128),(0, 0)), 'constant', constant_values=0)
                input = palette_texture_pad_2[640-256:640+256, i*256:i*256+512,:]
                if i == 3:
                    input[:,384:,:] = palette_texture_pad_2[640-256:640+256, 128:256, :]
#                 PIL.Image.fromarray(input, 'RGB').save(f'{outdir}/{os.path.basename(ipath)}_2_{i}_input.png')

                mask = mask_generator(input)
                if mask.min() == 1:
                    continue
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

                input = input.transpose(2, 0, 1) # HWC => CHW
                input = input[:3]
                input = (torch.from_numpy(input.copy()).float().to(device) / 127.5 - 1).unsqueeze(0)

                z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
                output = G(input, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                output = output[0].cpu().numpy()
                palette_texture_2[512-96:512+96, i*256:(i+1)*256,:] = output[256-96:256+96,256-128:256+128,:]

            PIL.Image.fromarray(palette_texture_2, 'RGB').save(f'{outdir}/{os.path.basename(ipath)}_fin.png')
            
if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
