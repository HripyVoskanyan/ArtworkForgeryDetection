import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def visualize_errors(misclassified_samples, images, save_dir="../Results/Error_Images"):
    """
    Visualizes and saves the first 10 misclassified samples.

    :param misclassified_samples: List of misclassified samples with image indices, true labels, predicted labels, and confidence scores.
    :param images: Dataset of images (tensor format) corresponding to misclassified_samples.
    :param save_dir: Directory to save the misclassified images.
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, sample in enumerate(misclassified_samples[:10]):  # Limit to 10 errors
        image_idx = sample["image_index"]
        true_label = sample["true_label"]
        predicted_label = sample["predicted_label"]
        confidence = sample["confidence"]

        # Convert tensor to PIL image and save it
        img = Image.fromarray((images[image_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"))
        img_path = os.path.join(save_dir, f"error_{i}_idx_{image_idx}.png")
        img.save(img_path)

        # Optionally, you can still save a plot for debugging or further use:
        plt.imshow(img)
        plt.title(f"True: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}")
        plt.axis("off")
        # Save the plot instead of showing it
        plt.savefig(os.path.join(save_dir, f"error_plot_{i}_idx_{image_idx}.png"))
        plt.close()  # Close the plot to free memory

    print(f"Misclassified samples visualized and saved to {save_dir}")


def analyze_errors(misclassified_samples):
    """
    Analyzes misclassified samples and generates an error summary.

    :param misclassified_samples: List of misclassified samples with true and predicted labels and confidence scores.
    :return: A dictionary containing error summary and confidence statistics.
    """
    if not misclassified_samples:
        print("No misclassified samples to analyze.")
        return {
            "error_summary": [],
            "confidence_stats": {}
        }

    df = pd.DataFrame(misclassified_samples)

    # Check if required columns exist
    required_columns = ["true_label", "predicted_label", "confidence"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col} in misclassified samples")

    # Group by true and predicted labels
    error_summary = df.groupby(["true_label", "predicted_label"]).size().reset_index(name="count")

    # Describe confidence statistics
    confidence_stats = df["confidence"].describe()

    print("Error Analysis Summary:")
    print(error_summary)
    print("\nConfidence Stats of Misclassifications:")
    print(confidence_stats)

    return {
        "error_summary": error_summary.to_dict(orient="records"),
        "confidence_stats": confidence_stats.to_dict()
    }


def save_error_analysis(error_summary, output_path="../Results/error_analysis.json"):
    """
    Saves error analysis results to a JSON file.

    :param error_summary: Dictionary containing error summary and confidence statistics.
    :param output_path: File path to save the analysis results.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(error_summary, f, indent=4)
