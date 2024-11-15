from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class StableDiffusionMethods:
    def __init__(self):
        # Load the Stable Diffusion model from Hugging Face using the 'diffusers' library
        print("Initializing Stable Diffusion Pipeline...")
        model_name = "stabilityai/stable-diffusion-2"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(
                self.device)
            print("Stable Diffusion Pipeline loaded successfully.")
        except Exception as e:
            print(f"Error occurred while loading the Stable Diffusion model: {e}")
            raise

    def generate_image_from_prompt(self, prompt, height=128, width=128, num_inference_steps=5, guidance_scale=2):
        """
        Generates an image from a given text prompt using Stable Diffusion.

        Args:
        - prompt (str): The input text prompt.
        - height (int): Height of the generated image.
        - width (int): Width of the generated image.
        - num_inference_steps (int): Number of inference steps for generation.
        - guidance_scale (float): Guidance scale for controlling prompt adherence.

        Returns:
        - PIL.Image: The generated image.
        """
        print(f"Generating image for prompt: '{prompt}'")
        print(
            f"Parameters: height={height}, width={width}, num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale}")

        if torch.cuda.is_available():
            print("CUDA is available. Preparing pipeline for GPU...")
            self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing()  # Enable memory-efficient attention
            self.pipeline.unet.to(torch.float16)  # Use half precision for better performance on GPU
            print("Pipeline prepared for GPU usage.")

        try:
            print("Starting image generation...")
            result = self.pipeline(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            print("Image generation completed.")
            image = result.images[0]
            print("Image generated successfully.")
        except Exception as e:
            print(f"Error occurred during image generation: {e}")
            raise

        return image
