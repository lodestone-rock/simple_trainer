from diffusers.models import AutoencoderKL
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from einops import rearrange
import cv2
import time
from safetensors.torch import save_file, load_file
DEVICE = "cuda"

# hwc
image = cv2.imread("/home/lodestone/Pictures/c767584f484491db0c0a1f13ae978fb9.jpg").astype(np.float32)
# image = cv2.resize(image, (128, 128))
# nchw
image = rearrange(image, " h w c -> c h w")[None, ...]


def load_vae(vae_safetensors_path:str):
    vae = AutoencoderKL.from_pretrained(vae_safetensors_path, use_safetensors=True)
    vae.to(device=DEVICE, dtype=torch.bfloat16)

    return vae


def image_to_latent(nchw_np_array_image, vae, image_name:str, save_path:str="."):
    with torch.no_grad():
        image =  torch.tensor((nchw_np_array_image / 255 - 0.5) * 2).to(device=DEVICE, dtype=torch.bfloat16)
        latent_dist = vae.encode(image).latent_dist.mean
        data = {
            "image_shape": torch.tensor(image.shape),
            "sdxl_vae_mean": latent_dist
        }
        save_file(data, f"{save_path}/{image_name}.safetensors")








vae = load_vae("vae")

image_to_latent(image, vae, "test_2", ".")



print()

latent = load_file("test_2.safetensors")

image_decoded = vae.decode(latent["sdxl_vae_mean"].to("cuda")).sample
image_decoded = torch.clip(image_decoded, -1, 1)
# nchw -> hwc
image_decoded = rearrange(image_decoded[0], "c h w -> h w c")
# rescale and save as image
image_decoded = np.array(((image_decoded + 1 ) * 127.5).to("cpu").to(torch.uint8))

cv2.imwrite('output_image2.jpg', image_decoded)


print()