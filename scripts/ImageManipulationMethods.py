import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class ImageManipulationMethods:
    def apply_affine_transformation(self, image):
        """
        Applies an affine transformation (random warp) to an image.

        Args:
        - image (numpy.ndarray): The input image in RGB format.

        Returns:
        - numpy.ndarray: The transformed image in RGB format.
        """
        # Convert RGB to BGR since OpenCV uses BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get the image dimensions
        rows, cols, ch = image_bgr.shape

        # Define the points for affine transformation
        source_points = np.float32([[0, 0], [cols, 0], [0, rows]])
        destination_points = np.float32([[0, 0], [cols + 50, 50], [50, rows - 50]])

        # Get the affine transformation matrix
        transformation_matrix = cv2.getAffineTransform(source_points, destination_points)

        # Apply the affine transformation
        transformed_image_bgr = cv2.warpAffine(image_bgr, transformation_matrix, (cols, rows))

        # Convert BGR back to RGB to keep consistency with DeepLake's image format
        transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_BGR2RGB)

        return transformed_image_rgb

    def seamless_blend(self, background_image, foreground_image):
        """
        Blends the foreground image seamlessly with the background image.

        Args:
        - background_image (numpy.ndarray): The background image in RGB format.
        - foreground_image (numpy.ndarray): The foreground image in RGB format.

        Returns:
        - numpy.ndarray: The blended image in RGB format.
        """
        # Convert RGB to BGR for OpenCV operations
        background_bgr = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)
        foreground_bgr = cv2.cvtColor(foreground_image, cv2.COLOR_RGB2BGR)

        # Ensure foreground fits within the background
        background_h, background_w = background_bgr.shape[:2]
        foreground_h, foreground_w = foreground_bgr.shape[:2]

        if foreground_h > background_h or foreground_w > background_w:
            # Resize the foreground to fit within the background dimensions
            scale_factor = min(background_h / foreground_h, background_w / foreground_w)
            new_size = (int(foreground_w * scale_factor), int(foreground_h * scale_factor))
            foreground_bgr = cv2.resize(foreground_bgr, new_size, interpolation=cv2.INTER_AREA)

        # Update the dimensions of the resized foreground
        foreground_h, foreground_w = foreground_bgr.shape[:2]

        # Define the center for seamless blending
        center_coordinates = (background_w // 2, background_h // 2)

        # Create a mask for the foreground
        mask = 255 * np.ones(foreground_bgr.shape, foreground_bgr.dtype)

        # Ensure the foreground image does not exceed the boundaries of the background
        start_x = max(0, center_coordinates[0] - foreground_w // 2)
        start_y = max(0, center_coordinates[1] - foreground_h // 2)

        if start_x + foreground_w > background_w:
            start_x = background_w - foreground_w

        if start_y + foreground_h > background_h:
            start_y = background_h - foreground_h

        # Adjust the center of the foreground for seamless cloning
        adjusted_center = (start_x + foreground_w // 2, start_y + foreground_h // 2)

        # Perform seamless cloning
        blended_image_bgr = cv2.seamlessClone(
            foreground_bgr,
            background_bgr,
            mask,
            adjusted_center,
            cv2.NORMAL_CLONE
        )

        # Convert BGR back to RGB
        blended_image_rgb = cv2.cvtColor(blended_image_bgr, cv2.COLOR_BGR2RGB)

        return blended_image_rgb
