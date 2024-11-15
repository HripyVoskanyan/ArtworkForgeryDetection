import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

class AdversarialAttackMethods:
    def __init__(self):
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define path to save the model
        self.resnet_model_path = 'models/resnet18.pth'

        # Load a pre-trained ResNet-18 model
        print("Loading pre-trained ResNet-18 model...")
        self.model = self.load_resnet_model()

        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def load_resnet_model(self):
        """
        Loads the ResNet-18 model from the specified path. If the model file doesn't exist,
        it downloads the model, saves it, and returns it.

        Returns:
        - torch.nn.Module: The loaded ResNet-18 model.
        """
        # Ensure the models directory exists
        if not os.path.exists(os.path.dirname(self.resnet_model_path)):
            os.makedirs(os.path.dirname(self.resnet_model_path))

        # Load or download the model
        if not os.path.exists(self.resnet_model_path):
            print("ResNet-18 model not found. Downloading and saving...")
            model = resnet18(pretrained=True)
            torch.save(model.state_dict(), self.resnet_model_path)
        else:
            model = resnet18()
            model.load_state_dict(torch.load(self.resnet_model_path, map_location=self.device))

        model.to(self.device)
        model.eval()
        return model

    def fgsm_attack(self, image_bgr, epsilon=0.03):
        """
        Performs FGSM attack on an image.

        Args:
        - image_bgr (numpy.ndarray): The input image in BGR format (OpenCV).
        - epsilon (float): Perturbation factor. Default is 0.03.

        Returns:
        - numpy.ndarray: The perturbed image in BGR format.
        """
        # Convert the input image to a PIL image and preprocess it
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_tensor = self.load_image(image_pil)

        # Set a random label for demo purposes (or use a correct label if available)
        label = torch.tensor([np.random.randint(0, 999)]).to(self.device)

        # Set requires_grad attribute of the input image to True for gradient computation
        image_tensor.requires_grad = True

        # Forward pass
        output = self.model(image_tensor)

        # Calculate the loss
        loss = self.loss_fn(output, label)

        # Backward pass: compute gradients with respect to the image
        self.model.zero_grad()
        loss.backward()

        # Collect the gradients of the image
        data_grad = image_tensor.grad.data

        # Perform FGSM attack
        perturbed_image_tensor = self._fgsm_attack(image_tensor, epsilon, data_grad)

        # Convert the perturbed image tensor to a NumPy array and return it in BGR format
        perturbed_image = perturbed_image_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        perturbed_image_bgr = cv2.cvtColor((perturbed_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        return perturbed_image_bgr

    def load_image(self, image_pil, max_size=224):
        """
        Loads and preprocesses an image for ResNet.

        Args:
        - image_pil (PIL.Image): The input image in PIL format.
        - max_size (int): Size to resize the image to. Default is 224.

        Returns:
        - torch.Tensor: The preprocessed image tensor.
        """
        transform = transforms.Compose([
            transforms.Resize((max_size, max_size)),
            transforms.ToTensor(),
        ])
        return transform(image_pil).unsqueeze(0).to(self.device)

    def _fgsm_attack(self, image, epsilon, data_grad):
        """
        Helper function to perform FGSM attack.

        Args:
        - image (torch.Tensor): The input image tensor.
        - epsilon (float): Perturbation factor.
        - data_grad (torch.Tensor): The gradients of the image.

        Returns:
        - torch.Tensor: The perturbed image tensor.
        """
        # Collect the sign of the gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting the input image by epsilon along the gradient
        perturbed_image = image + epsilon * sign_data_grad
        # Clamp the values of the perturbed image to [0, 1] to keep it in a valid range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

