import os
import random
from UniversalFaker import UniversalFaker
import cv2
import deeplake
import numpy as np
from PIL import Image
import warnings
<<<<<<< HEAD
import re

warnings.filterwarnings("ignore")

def extract_numbers_from_filenames(folder_path):
    """
    Extract numbers from filenames in the given folder.
    """
    extracted_numbers = set()
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            numbers = re.findall(r'\d+', filename)
            if numbers:
                extracted_numbers.update(map(int, numbers))
    return extracted_numbers

def run_poisoning():
    # Initialize UniversalFaker
=======
warnings.filterwarnings("ignore")

def run_poisoning():
>>>>>>> parent of 6d28175 (Update .gitignore to exclude scripts, data/poisoned, and .venv)
    universal_faker = UniversalFaker()
    deeplake_dataset = 'hub://activeloop/wiki-art'

    # Load dataset
    ds = deeplake.load(deeplake_dataset)

<<<<<<< HEAD
=======
    # Select random subset of images for transformation (e.g., 1)
    random_indices = random.sample(range(len(ds)), 2)

>>>>>>> parent of 6d28175 (Update .gitignore to exclude scripts, data/poisoned, and .venv)
    # Output folder for fake images
    output_folder = "../data/poisoned"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

<<<<<<< HEAD
    # Extract numbers from existing filenames in the poisoned folder
    existing_numbers = extract_numbers_from_filenames(output_folder)

    # Generate random indices while avoiding existing numbers
    random_indices = []
    while len(random_indices) < 5000:  # Adjust number of samples as needed
        idx = random.randint(0, len(ds) - 1)
        if idx not in existing_numbers:
            random_indices.append(idx)
            existing_numbers.add(idx)  # Add to the set to avoid duplicates

=======
>>>>>>> parent of 6d28175 (Update .gitignore to exclude scripts, data/poisoned, and .venv)
    # Apply transformations
    for idx in random_indices:
        sample = ds[idx]
        original_image = sample['images'].numpy()  # Get as numpy array

        # Apply random transformation using UniversalFaker
<<<<<<< HEAD
        fake_image, method_name = universal_faker.apply_random_transformation(original_image)
=======
        fake_image, method_name = universal_faker.apply_random_transformation(original_image)[0], universal_faker.apply_random_transformation(original_image)[1]
>>>>>>> parent of 6d28175 (Update .gitignore to exclude scripts, data/poisoned, and .venv)

        # If the fake_image is a PIL Image, convert it to a NumPy array
        if isinstance(fake_image, Image.Image):
            fake_image = np.array(fake_image)

        # Convert from RGB (PIL uses RGB) to BGR (OpenCV uses BGR) if needed
        if len(fake_image.shape) == 3 and fake_image.shape[2] == 3:
            fake_image = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)

        # Save the transformed image
        output_path = os.path.join(output_folder, f"{method_name}_{idx}.png")
        cv2.imwrite(output_path, fake_image)
        print(f"Saved transformed image to: {output_path}")

if __name__ == "__main__":
    run_poisoning()
