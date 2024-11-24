import os
import random
from PIL import Image
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

random.seed(777)

def create_combined_dataset(fake_path, real_path, target_size=(256, 256), real_limit=10000, fake_limit=10000):
    """
    This function loads a specified number of real and fake images from local directories and prepares the dataset for training.
    It will randomly select the specified number of images from each folder.

    :param fake_path: Path to the directory containing fake images.
    :param real_path: Path to the directory containing real images.
    :param target_size: Target size to resize the images.
    :param real_limit: The exact number of real images to randomly select.
    :param fake_limit: The exact number of fake images to randomly select.
    :return: images (list of transformed images), labels (list of corresponding labels)
    """

    # Load real images from the local directory
    real_images = []
    real_files = [f for f in os.listdir(real_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    real_files = random.sample(real_files, min(real_limit, len(real_files)))  # Randomly select the number of images
    for img_file in real_files:
        with Image.open(os.path.join(real_path, img_file)) as img:
            real_images.append(img.copy())  # Copy image to close the file handle
    print(f'{len(real_images)} real images loaded')

    # Load fake images from the local directory
    fake_images = []
    fake_files = [f for f in os.listdir(fake_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    fake_files = random.sample(fake_files, min(fake_limit, len(fake_files)))  # Randomly select the number of images
    for img_file in fake_files:
        with Image.open(os.path.join(fake_path, img_file)) as img:
            fake_images.append(img.copy())  # Copy image to close the file handle
    print(f'{len(fake_images)} fake images loaded')

    # Define transformation (resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    # Apply transformation to all images
    real_images = [transform(img) for img in real_images]
    fake_images = [transform(img) for img in fake_images]
    print('Images transformed')

    # Create labels: 0 for real images, 1 for fake images
    real_labels = [0] * len(real_images)
    fake_labels = [1] * len(fake_images)

    # Combine the images and labels
    images = real_images + fake_images
    labels = real_labels + fake_labels

    return images, labels
