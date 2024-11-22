import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class StyleTransferMethods:
    def __init__(self):
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def apply_style_transfer(self, content_image, style_image, num_steps=500, content_weight=1e4, style_weight=1e2):
        # Load VGG19 model
        vgg = models.vgg19(pretrained=True).features.to(self.device).eval()
        for param in vgg.parameters():
            param.requires_grad_(False)

        # Load content and style images (assuming they are numpy arrays)
        content_tensor = self.load_image(content_image)
        style_tensor = self.load_image(style_image)

        # Extract content and style features
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        content_features = self.get_features(content_tensor, vgg, content_layers)
        style_features = self.get_features(style_tensor, vgg, style_layers)

        # Start with a copy of the content image as the "target"
        target = content_tensor.clone().requires_grad_(True).to(self.device)
        optimizer = optim.Adam([target], lr=0.003)

        # Style weights for each layer
        style_weights = {layer: 0.2 for layer in style_layers}

        # Optimization loop
        for step in range(1, num_steps + 1):
            target_features = self.get_features(target, vgg, style_layers + content_layers)
            content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4']) ** 2)

            # Style loss
            style_loss = 0
            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                style_gram = self.gram_matrix(style_features[layer])
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_style_loss

            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss

            # Update target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Post-process and return the stylized image
        stylized_image_pil = self.unnormalize(target)

        # Convert PIL image to NumPy array (BGR format for OpenCV)
        stylized_image_np = np.array(stylized_image_pil)
        stylized_image_bgr = cv2.cvtColor(stylized_image_np, cv2.COLOR_RGB2BGR)

        return stylized_image_bgr

    def load_image(self, image, max_size=400):
        if isinstance(image, str):
            # If the input is a path, load the image from the path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # If the input is a NumPy array, convert it to a PIL Image
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        transform = transforms.Compose([
            transforms.Resize((max_size, max_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def get_features(self, image, model, layers):
        features = {}
        x = image
        for name, layer in enumerate(model):
            x = layer(x)
            if f'conv_{name + 1}' in layers:
                features[f'conv_{name + 1}'] = x
        return features

    def gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram

    def unnormalize(self, tensor):
        tensor = tensor.clone().detach()
        tensor = tensor.cpu().squeeze(0)  # Remove batch dimension
        tensor = tensor * torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1) + torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)  # Unnormalize
        tensor = tensor.clamp(0, 1)  # Clamp the values between 0 and 1
        return transforms.ToPILImage()(tensor)

    def enhance_image_color(self, image, factor=1.5):
        """
        Enhances the color of an image and returns the result.

        Args:
        - image (numpy.ndarray): The input image (BGR format).
        - factor (float): The enhancement factor for color. Default is 1.5.

        Returns:
        - numpy.ndarray: The enhanced image (BGR format).
        """
        # Convert the image from BGR (OpenCV format) to RGB (PIL format)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        pil_img = Image.fromarray(img_rgb)

        # Enhance the color using PIL's ImageEnhance
        enhancer = ImageEnhance.Color(pil_img)
        enhanced_img = enhancer.enhance(factor)

        # Convert back to OpenCV format (RGB to BGR)
        enhanced_img_bgr = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)

        return enhanced_img_bgr

    def apply_photorealistic_style_transfer(self, content_image, style_image):
        """
        Apply photorealistic style transfer to an image.

        Args:
        - content_image (numpy.ndarray): The input content image (BGR format).
        - style_image (numpy.ndarray): The style image to be transferred (BGR format).

        Returns:
        - numpy.ndarray: The photorealistically stylized image (BGR format).
        """
        # Convert images from BGR to RGB and to PIL format
        content_image_rgb = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
        style_image_rgb = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
        content_pil = Image.fromarray(content_image_rgb)
        style_pil = Image.fromarray(style_image_rgb)

        # Load the content and style images
        content_tensor = self.load_image(content_pil)
        style_tensor = self.load_image(style_pil)

        # Extract content features
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        content_features = self.get_features(content_tensor, vgg, content_layers)
        style_features = self.get_features(style_tensor, vgg, style_layers)

        # Start with a copy of the content image as the "target"
        target = content_tensor.clone().requires_grad_(True).to(self.device)
        optimizer = optim.Adam([target], lr=0.003)

        # Style weights for each layer
        style_weights = {layer: 0.2 for layer in style_layers}

        # Optimization loop
        for step in range(1, 101):
            target_features = self.get_features(target, vgg, style_layers + content_layers)
            content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4'])**2)

            # Style loss
            style_loss = 0
            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                style_gram = self.gram_matrix(style_features[layer])
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                style_loss += layer_style_loss

            # Total loss
            total_loss = content_loss + 1e-2 * style_loss

            # Update target image
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

            # Print progress
            if step % 10 == 0:
                print(f"Step {step}, Total Loss: {total_loss.item()}")

        # Post-process and return the stylized image
        output_image = target.squeeze().cpu().detach().numpy()
        output_image = (output_image * 0.229 + 0.485).clip(0, 1)  # Unnormalize
        output_image = (output_image * 255).astype(np.uint8)
        output_image = np.transpose(output_image, (1, 2, 0))
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        return output_image_bgr
