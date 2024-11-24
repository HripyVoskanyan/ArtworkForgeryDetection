from PIL import Image
import os

def create_combined_dataset(real_path, fake_path):
    dataset = []
    labels = []
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}  # Add more extensions if needed

    # Process real images
    for img_file in os.listdir(real_path):
        file_ext = os.path.splitext(img_file)[1].lower()  # Get file extension
        if file_ext not in allowed_extensions:
            print(f"Skipping non-image file: {img_file}")
            continue

        try:
            with Image.open(os.path.join(real_path, img_file)) as img:
                img = img.convert("RGB")  # Ensure it's in RGB format
                dataset.append(img)
                labels.append(0)  # Label 0 for real images
        except Exception as e:
            print(f"Error processing file {img_file}: {e}")
            continue

    # Process fake images
    for img_file in os.listdir(fake_path):
        file_ext = os.path.splitext(img_file)[1].lower()
        if file_ext not in allowed_extensions:
            print(f"Skipping non-image file: {img_file}")
            continue

        try:
            with Image.open(os.path.join(fake_path, img_file)) as img:
                img = img.convert("RGB")
                dataset.append(img)
                labels.append(1)  # Label 1 for fake images
        except Exception as e:
            print(f"Error processing file {img_file}: {e}")
            continue

    return dataset, labels
