from PIL import Image, ImageEnhance
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class ColorStyleMethods:
    def __init__(self):
        pass

    def enhance_image_color(self, image, factor=1.5):
        """
        Enhances the color of an image.

        Args:
        - image (numpy.ndarray): The input image in BGR format (OpenCV).
        - factor (float): The enhancement factor for color. Default is 1.5.

        Returns:
        - numpy.ndarray: The enhanced image in BGR format.
        """
        # Convert the image from BGR (OpenCV format) to RGB (PIL format)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        pil_img = Image.fromarray(img_rgb)

        # Enhance the color using PIL's ImageEnhance
        enhancer = ImageEnhance.Color(pil_img)
        enhanced_img = enhancer.enhance(factor)

        # Convert back to NumPy array (RGB)
        enhanced_img_np = np.array(enhanced_img)

        # Convert back to BGR format for OpenCV compatibility
        enhanced_img_bgr = cv2.cvtColor(enhanced_img_np, cv2.COLOR_RGB2BGR)

        return enhanced_img_bgr

    def change_color_palette(self, image, palette_type='sepia'):
        """
        Changes the color palette of the image.

        Args:
        - image (numpy.ndarray): The input image in BGR format (OpenCV).
        - palette_type (str): The type of palette to apply (e.g., 'sepia').

        Returns:
        - numpy.ndarray: The image with the modified color palette.
        """
        if palette_type == 'sepia':
<<<<<<< HEAD
            image = image.astype('float32')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
=======
>>>>>>> parent of 6d28175 (Update .gitignore to exclude scripts, data/poisoned, and .venv)
            # Create a filter for sepia effect
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            sepia_image = cv2.transform(image, kernel)
            sepia_image = np.clip(sepia_image, 0, 255)
            return sepia_image.astype(np.uint8)

        elif palette_type == 'grayscale':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

<<<<<<< HEAD
=======
        # Add more palette types as needed...

>>>>>>> parent of 6d28175 (Update .gitignore to exclude scripts, data/poisoned, and .venv)
        return image