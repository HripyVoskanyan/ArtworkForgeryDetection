import os
import shutil

# Paths
source_folder = '../data/wikiart'  # Adjust the path to point to the correct folder relative to 'scripts'
destination_folder = '../data/originals'  # Adjusted accordingly

# Create destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate through all genre folders in the source folder
for genre_folder in os.listdir(source_folder):
    genre_path = os.path.join(source_folder, genre_folder)

    if os.path.isdir(genre_path):  # Only process folders
        # Move all files from genre folder to the destination folder
        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            destination_path = os.path.join(destination_folder, file_name)

            if os.path.isfile(file_path):  # Only move files, not subdirectories
                # Check if the file already exists in the destination folder
                if os.path.exists(destination_path):
                    print(f"File {file_name} already exists, skipping.")
                    continue  # Skip this file if it already exists

                # Move the file to the destination folder
                shutil.move(file_path, destination_folder)

        # Remove the folder even if it's not empty (e.g., if hidden files exist)
        shutil.rmtree(genre_path)

print("All images have been moved and folders deleted.")
