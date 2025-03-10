import torch
import numpy as np
import random
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from StyleStudio.ip_adapter.utils import BLOCKS
from StyleStudio.ip_adapter import StyleStudio_Adapter
from comfy.model_management import get_torch_device


device = get_torch_device()


class StyleStudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "image_encoder_path": ("STRING", {"default": "h94/IP-Adapter/sdxl_models/image_encoder"}),
                "adapter_checkpoint": ("STRING", {"default": "InstantX/CSGO/csgo_4_32.bin"}),
                "vae_path": ("STRING", {"default": "madebyollin/sdxl-vae-fp16-fix"}),
                "style_image_path": ("STRING", {"default": "assets/style_image.jpg"}),
                "prompt": ("STRING", {"default": "A red apple"}),
                "negative_prompt": ("STRING", {"default": "text, watermark, lowres, deformed, blurry"}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 15.0, "step": 0.1}),
                "num_steps": ("INT", {"default": 50, "min": 5, "max": 200, "step": 1}),
                "end_fusion": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "cross_modal_adain": ("BOOLEAN", {"default": True}),
                "use_sattn": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": np.iinfo(np.int32).max}),
                "randomize_seed": ("BOOLEAN", {"default": False}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "StyleStudio"

    def __init__(self):
        self.pipe = None
        self.adapter = None

    def load_pipeline(self, model_path, image_encoder_path, adapter_checkpoint, vae_path):
        if self.pipe is None:
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                add_watermarker=False,
                vae=vae
            )
            self.pipe.enable_vae_tiling()

        if self.adapter is None:
            target_style_blocks = BLOCKS["style"]
            self.adapter = StyleStudio_Adapter(
                self.pipe, image_encoder_path, adapter_checkpoint, device, num_style_tokens=32,
                target_style_blocks=target_style_blocks,
                controlnet_adapter=False,
                style_model_resampler=True,
                fuSAttn=True,
                end_fusion=20,
                adainIP=True
            )

    def generate_image(self, model_path, image_encoder_path, adapter_checkpoint, vae_path,
                       style_image_path, prompt, negative_prompt, guidance_scale, num_steps, 
                       end_fusion, cross_modal_adain, use_sattn, seed, randomize_seed, height, width):

        self.load_pipeline(model_path, image_encoder_path, adapter_checkpoint, vae_path)

        if randomize_seed:
            seed = random.randint(0, np.iinfo(np.int32).max)
        
        print(f"Using seed: {seed}")
        generator = torch.Generator(device).manual_seed(seed)
        init_latents = torch.randn((1, 4, height // 8, width // 8), generator=generator, device=device, dtype=torch.float16)

        num_sample = 1
        if use_sattn:
            num_sample = 2
            init_latents = init_latents.repeat(num_sample, 1, 1, 1)

        style_image = Image.open(style_image_path).convert("RGB").resize((width, height))

        with torch.no_grad():
            images = self.adapter.generate(
                pil_style_image=style_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                num_samples=num_sample,
                num_inference_steps=num_steps,
                end_fusion=end_fusion,
                cross_modal_adain=cross_modal_adain,
                use_SAttn=use_sattn,
                generator=generator,
                latents=init_latents
            )

        output_image = np.array(images[1] if use_sattn else images[0], dtype=np.uint8)
        return (output_image,)

