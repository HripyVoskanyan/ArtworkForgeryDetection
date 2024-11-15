from NoiseMethods import NoiseMethods
from AdversarialAttackMethods import AdversarialAttackMethods
from ColorStyleMethods import ColorStyleMethods
from ImageManipulationMethods import ImageManipulationMethods
from StyleTransferMethods import StyleTransferMethods
from DeepDreamMethods import DeepDreamMethods
import deeplake
import random
import warnings
warnings.filterwarnings("ignore")

class UniversalFaker:
    def __init__(self):
        self.noise_methods = NoiseMethods()
        self.adversarial_methods = AdversarialAttackMethods()
        self.color_methods = ColorStyleMethods()
        self.image_manipulation_methods = ImageManipulationMethods()
        self.style_transfer_methods = StyleTransferMethods()
        self.deepdream_methods = DeepDreamMethods()
    def apply_random_transformation(self, image):
        """
        Apply a randomly chosen transformation to the image.
        Expensive transformations will have lower probability.

        Args:
        - image (numpy.ndarray): The input image.

        Returns:
        - numpy.ndarray: The transformed image.
        """
        transformations = [
            #("add_gaussian_noise", self.noise_methods.add_gaussian_noise, 0.3),
            #("add_salt_and_pepper_noise", self.noise_methods.add_salt_and_pepper_noise, 0.3),
            #("enhance_image_color", self.color_methods.enhance_image_color, 0.3),
            #('change_color_palette', self.color_methods.change_color_palette, 0.3),
            #("apply_deepdream", self.deepdream_methods.apply_deepdream, 0.15),
            #("fgsm_attack", self.adversarial_methods.fgsm_attack, 0.15),
            #("seamless_blend", self.image_manipulation_methods.seamless_blend, 0.1),
            ("apply_style_transfer", self.style_transfer_methods.apply_style_transfer, 0.1),
            #("apply_photorealistic_style_transfer", self.style_transfer_methods.apply_photorealistic_style_transfer, 0.1),

        ]

        # Select a transformation based on probability
        weights = [transformation[2] for transformation in transformations]
        selected_transformation = random.choices(transformations, weights, k=1)[0]

        # Apply the selected transformation
        method_name, method_function, _ = selected_transformation
        if method_name == "apply_style_transfer" or method_name == "apply_photorealistic_style_transfer":
            # Style transfer requires two images
            return method_function(image, self.get_random_style_image()), method_name
        elif method_name == "seamless_blend":
            # Blending requires two images
            return method_function(image, self.get_random_foreground_image()), method_name
        elif method_name == "generate_stable_diffusion_image":
            # Stable Diffusion requires a prompt
            prompt = "A beautiful painting of a sunset over a mountain range"
            return method_function(prompt), method_name
        else:
            return method_function(image), method_name

    def get_random_style_image(self):
        """
        Get a random style image for style transfer.
        """
        # Add logic to load a random style image from DeepLake dataset
        # Replace this with actual loading from your DeepLake dataset
        return self.load_random_image_from_dataset()

    def get_random_foreground_image(self):
        """
        Get a random foreground image for blending.
        """
        # Add logic to load a random foreground image from DeepLake dataset
        # Replace this with actual loading from your DeepLake dataset
        return self.load_random_image_from_dataset()

    def load_random_image_from_dataset(self):
        """
        Load a random image from the dataset.
        This should be replaced with a method to load from DeepLake.
        """
        ds = deeplake.load('hub://activeloop/wiki-art')
        idx = random.randint(0, len(ds) - 1)
        sample = ds[idx]
        return sample['images'].numpy()  # Return as numpy array
