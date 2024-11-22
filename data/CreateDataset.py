import deeplake
from PIL import Image
import os
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

def create_combined_dataset(fake_path, target_size=(256, 256), real_limit=50, fake_limit=15):
    # Load WikiArt dataset from DeepLake
    ds = deeplake.load("hub://activeloop/wiki-art")
    real_images = []
    for i in range(min(real_limit, len(ds['images']))):
        img = ds['images'][i].numpy()
        real_images.append(Image.fromarray(img))
    print('Real images loaded')

    # Load fake images from local directory
    fake_images = []
    fake_files = [f for f in os.listdir(fake_path) if f.endswith(('png', 'jpg', 'jpeg'))][:fake_limit]
    for img_file in fake_files:
        with Image.open(os.path.join(fake_path, img_file)) as img:
            fake_images.append(img.copy())  # Copy image to close the file handle
    print('Fake images loaded')
    print(len(fake_images))

    # Define transformation
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    # Apply transformation
    real_images = [transform(img) for img in real_images]
    fake_images = [transform(img) for img in fake_images]
    print('Images transformed')

    # Create labels
    real_labels = [0] * len(real_images)
    fake_labels = [1] * len(fake_images)

    # Combine datasets
    images = real_images + fake_images
    labels = real_labels + fake_labels

    return images, labels
