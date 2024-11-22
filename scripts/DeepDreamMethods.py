import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class DeepDreamMethods:
    def __init__(self):
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the pretrained InceptionV3 model
        self.model = models.inception_v3(pretrained=True).to(self.device).eval()

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Register hooks for intermediate layers
        self.activations = None
        layer_name = 'Mixed_5b'  # Example layer for deep dreaming effect
        layer = dict(self.model.named_modules())[layer_name]
        layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        """Hook function to capture layer activations."""
        self.activations = output

    def apply_deepdream(self, image, iterations=100, lr=0.02, max_size=300, intensity=1.5):
        """
        Applies DeepDream effect to an image with a noticeable effect.

        Args:
        - image (numpy.ndarray): The input image in NumPy format (H, W, C).
        - iterations (int): Number of iterations for deep dreaming. Default is 15.
        - lr (float): Learning rate for gradient ascent. Default is 0.02.
        - max_size (int): Resize image to this max dimension for faster processing.
        - intensity (float): Factor to scale the loss, increasing effect intensity.

        Returns:
        - numpy.ndarray: The DeepDream image in NumPy format.
        """
        input_tensor = self._load_image_from_numpy(image, max_size=max_size).to(self.device)
        input_tensor.requires_grad = True

        for _ in range(iterations):
            self.model(input_tensor)
            loss = self.activations.norm() * intensity  # Scale the loss for stronger effect

            self.model.zero_grad()
            loss.backward()
            input_tensor.data += lr * input_tensor.grad.data
            input_tensor.grad.data.zero_()

        output_image = input_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        output_image = np.clip((output_image * 255), 0, 255).astype(np.uint8)
        return output_image

    def _load_image_from_numpy(self, image_numpy, max_size=300):
        """
        Loads and preprocesses an image from NumPy array.

        Args:
        - image_numpy (numpy.ndarray): The input image in NumPy format (H, W, C).
        - max_size (int): Resize the image to this max dimension for faster processing.

        Returns:
        - torch.Tensor: The preprocessed image tensor.
        """
        # Convert NumPy array to PIL Image
        image_pil = Image.fromarray(image_numpy)

        # Define transformations: resize, convert to tensor, and normalize
        transform = transforms.Compose([
            transforms.Resize((max_size, max_size)),
            transforms.ToTensor(),
        ])

        return transform(image_pil).unsqueeze(0)


