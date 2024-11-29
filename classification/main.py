import os
import sys
import itertools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.CreateDataset import create_combined_dataset  # Import your dataset creation function
from analyze_errors import visualize_errors, analyze_errors, save_error_analysis
from ResNetClassifier import ResNetClassifier
from ViTClassifier import ViTClassifier
from CLIPViTClassifier import ViTCLIPPipeline, FocalLoss, ContrastiveLoss, TripletLoss, CLIPViTClassifier, EmbeddingDataset as CLIPEmbeddingDataset
from DINOClassifier import DINOEmbeddingPipeline, DINOClassifier, EmbeddingDataset as DINOEmbeddingDataset
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation
from early_stopping import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert tensor to PIL image if necessary
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)

        # Apply the transform if provided
        if self.transform:
            image = self.transform(image)

        return image, label


# Create necessary directories
os.makedirs("../Results", exist_ok=True)
os.makedirs("../models", exist_ok=True)

# Load dataset
print("Loading dataset...")
images, labels = create_combined_dataset(real_path='../data/originals', fake_path="../data/poisoned", real_limit=15000, fake_limit=15000)

# Split dataset
print("Splitting dataset...")
images_train, images_temp, labels_train, labels_temp = train_test_split(
    images, labels, test_size=0.4, stratify=labels, random_state=42
)
images_val, images_test, labels_val, labels_test = train_test_split(
    images_temp, labels_temp, test_size=0.5, stratify=labels_temp, random_state=42
)

batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet and ViT (general CNNs)
resnet_transform = Compose([
    Resize((224, 224)),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])
#CLIP (ViT-B/32 specific normalization)
clip_transform = Compose([
    Resize(224),
    CenterCrop(224),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    ToTensor(),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# DINO (ViT-B/16 specific normalization)
dino_transform = Compose([
    Resize((224, 224)),
    CenterCrop(224),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# # Define ViT preprocessing
vit_transform = Compose([
    Resize((224, 224)),            # Resize to 224x224 (ViT input size)
    CenterCrop(224),               # Crop the image to 224x224 (ViT input size)
    ToTensor(),                    # Convert PIL image to tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet mean & std)
])

# -------------------
# ViT+CLIP Classifier
# -------------------
print("\nTraining ViT+CLIP Classifier...")


# Define CLIP preprocessing

clip_pipeline = ViTCLIPPipeline(device=device)

# Update DataLoaders for CLIP with the correct transform
clip_train_loader = DataLoader(ImageDataset(images_train, labels_train, transform=clip_transform), batch_size=1, shuffle=False)
clip_val_loader = DataLoader(ImageDataset(images_val, labels_val, transform=clip_transform), batch_size=1, shuffle=False)
clip_test_loader = DataLoader(ImageDataset(images_test, labels_test, transform=clip_transform), batch_size=1, shuffle=False)

clip_train_embeddings, clip_train_labels = clip_pipeline.extract_clip_embeddings(clip_train_loader)
clip_val_embeddings, clip_val_labels = clip_pipeline.extract_clip_embeddings(clip_val_loader)
clip_test_embeddings, clip_test_labels = clip_pipeline.extract_clip_embeddings(clip_test_loader)

clip_classifier = CLIPViTClassifier(embedding_dim=512).to(device)
clip_train_dataset = CLIPEmbeddingDataset(clip_train_embeddings, clip_train_labels)
clip_val_dataset = CLIPEmbeddingDataset(clip_val_embeddings, clip_val_labels)
clip_test_dataset = CLIPEmbeddingDataset(clip_test_embeddings, clip_test_labels)

clip_train_loader = DataLoader(clip_train_dataset, batch_size=64, shuffle=True)
clip_val_loader = DataLoader(clip_val_dataset, batch_size=64, shuffle=False)
clip_test_loader = DataLoader(clip_test_dataset, batch_size=64, shuffle=False)

# search_space = {
#     'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
#     'alpha': [0.5, 0.75, 1.0],
#     'gamma': [1, 2, 3],
#     'dropout_rate': [0.3, 0.5, 0.6],
#     'weight_decay': [1e-5, 1e-4, 1e-3],
#     'batch_size': [16, 32, 64],
# }
# grid = list(itertools.product(*search_space.values()))
# best_model = None
# best_accuracy = 0
# best_hyperparams = {}
criterion = FocalLoss(alpha=0.75, gamma=1)
#criterion = ContrastiveLoss(margin=0.1)
#criterion = TripletLoss(margin=0.1)
# Initialize the optimizer
optimizer = torch.optim.Adam(clip_classifier.parameters(), lr=0.005, weight_decay=0.0001)

# Train the classifier
clip_train_results = clip_pipeline.train_classifier(
    clip_classifier,
    clip_train_loader,
    clip_val_loader,
    criterion,  # Use FocalLoss
    optimizer,
    epochs=30)
# for params in grid:
#     # Unpack hyperparameters
#     learning_rate, alpha, gamma, dropout_rate, weight_decay, batch_size = params
#
#     # Create DataLoaders with the current batch size
#     clip_train_loader = DataLoader(clip_train_dataset, batch_size=batch_size, shuffle=True)
#     clip_val_loader = DataLoader(clip_val_dataset, batch_size=batch_size, shuffle=False)
#     clip_test_loader = DataLoader(clip_test_dataset, batch_size=batch_size, shuffle=False)
#
#     # Initialize the model with the current dropout rate
#     clip_classifier = CLIPViTClassifier(embedding_dim=512, dropout_rate=dropout_rate).to(device)
#
#     # Define the loss function and optimizer
#     criterion = FocalLoss(alpha=alpha, gamma=gamma)
#     optimizer = torch.optim.AdamW(
#         clip_classifier.parameters(),
#         lr=learning_rate,
#         weight_decay=weight_decay
#     )
#
#     # Train the model with the current hyperparameters
#     vit_clip_pipeline = ViTCLIPPipeline()
#     results = vit_clip_pipeline.train_classifier(
#         model=clip_classifier,
#         train_loader=clip_train_loader,
#         val_loader=clip_val_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         epochs=20  # Adjust as needed
#     )
#
#     # Evaluate the model on the validation set
#     val_accuracy = max(epoch['val_accuracy'] for epoch in results['epochs'])
#     if val_accuracy > best_accuracy:
#         best_accuracy = val_accuracy
#         best_model = clip_classifier
#         best_hyperparams = {
#             'learning_rate': learning_rate,
#             'alpha': alpha,
#             'gamma': gamma,
#             'dropout_rate': dropout_rate,
#             'weight_decay': weight_decay,
#             'batch_size': batch_size,
#         }
#
# # Save the best model and hyperparameters
# print("Best Hyperparameters:", best_hyperparams)
# print("Best Validation Accuracy:", best_accuracy)
#vit_clip_pipeline.save_model(best_model, "../models/best_vit_clip_classifier.pth")

#Evaluate the classifier
clip_test_results, clip_misclassified_samples = clip_pipeline.evaluate_classifier(clip_classifier, clip_test_loader)

#Analyze and visualize errors
visualize_errors(clip_misclassified_samples, images_test)

clip_error_summary = analyze_errors(clip_misclassified_samples)
if clip_error_summary["error_summary"]:
    save_error_analysis(clip_error_summary, output_path="../Results/error_analysis_vit_clip.json")
else:
    print("No errors to save for ViT+CLIP classifier.")
clip_pipeline.save_results(
    {"train_results": clip_train_results, "test_results": clip_test_results},
    output_path="../Results/vit_clip_results.json",
)
clip_pipeline.save_model(clip_classifier, model_path="../models/vit_and_clip_classifier.pth")

# -------------------
# DINO Classifier
# -------------------
print("\nTraining DINO Classifier...")

# Initialize the DINO pipeline
dino_pipeline = DINOEmbeddingPipeline(device=device)

# Create DataLoaders for DINO with proper transformations
dino_train_loader = DataLoader(
    ImageDataset(images_train, labels_train, transform=dino_transform),
    batch_size=1, shuffle=False
)
dino_val_loader = DataLoader(
    ImageDataset(images_val, labels_val, transform=dino_transform),
    batch_size=1, shuffle=False
)
dino_test_loader = DataLoader(
    ImageDataset(images_test, labels_test, transform=dino_transform),
    batch_size=1, shuffle=False
)

# Extract embeddings using DINO pipeline
dino_train_embeddings, dino_train_labels = dino_pipeline.extract_embeddings(dino_train_loader)
dino_val_embeddings, dino_val_labels = dino_pipeline.extract_embeddings(dino_val_loader)
dino_test_embeddings, dino_test_labels = dino_pipeline.extract_embeddings(dino_test_loader)

# Create embedding datasets for training and validation
dino_train_dataset = DINOEmbeddingDataset(dino_train_embeddings, dino_train_labels)
dino_val_dataset = DINOEmbeddingDataset(dino_val_embeddings, dino_val_labels)
dino_test_dataset = DINOEmbeddingDataset(dino_test_embeddings, dino_test_labels)

# Create DataLoaders for training and evaluation
dino_train_loader = DataLoader(dino_train_dataset, batch_size=batch_size, shuffle=True)
dino_val_loader = DataLoader(dino_val_dataset, batch_size=batch_size, shuffle=False)
dino_test_loader = DataLoader(dino_test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the DINO classifier model
dino_classifier = DINOClassifier(embedding_dim=768).to(device)

# Train the DINO classifier
dino_train_results = dino_pipeline.train_classifier(
    dino_classifier,
    dino_train_loader,
    dino_val_loader,
    nn.BCEWithLogitsLoss(),
    torch.optim.Adam(dino_classifier.parameters(), lr=1e-4, weight_decay=1e-5),
    epochs=30
)

# Evaluate the DINO classifier
dino_test_results, dino_misclassified_samples = dino_pipeline.evaluate_classifier(dino_classifier, dino_test_loader)

# Analyze and visualize errors
if dino_misclassified_samples:
    visualize_errors(dino_misclassified_samples, images_test)
    dino_error_summary = analyze_errors(dino_misclassified_samples)
    save_error_analysis(dino_error_summary, output_path="../Results/error_analysis_dino.json")
else:
    print("No misclassified samples to analyze.")

# Save results
dino_pipeline.save_results(
    {"train_results": dino_train_results, "test_results": dino_test_results},
    output_path="../Results/dino_results.json"
)
dino_pipeline.save_model(dino_classifier, model_path="../models/dino_classifier.pth")



# -------------------
# ResNet Classifier
# -------------------
print("\nTraining ResNet Classifier...")
resnet = ResNetClassifier(device=device)
# Create DataLoaders for ResNet with proper transformations
resnet_train_loader = DataLoader(
    ImageDataset(images_train, labels_train, transform=resnet_transform),
    batch_size=32, shuffle=True
)
resnet_val_loader = DataLoader(
    ImageDataset(images_val, labels_val, transform=resnet_transform),
    batch_size=32, shuffle=False
)
resnet_test_loader = DataLoader(
    ImageDataset(images_test, labels_test, transform=resnet_transform),
    batch_size=32, shuffle=False
)

resnet_train_results = resnet.train_with_validation(resnet_train_loader, resnet_val_loader, epochs=20)
resnet_test_results, resnet_misclassified_samples = resnet.evaluate_model(resnet_test_loader)

if resnet_misclassified_samples:
    visualize_errors(resnet_misclassified_samples, images_test)
    resnet_error_summary = analyze_errors(resnet_misclassified_samples)
    save_error_analysis(resnet_error_summary, output_path="../Results/error_analysis_resnet.json")
else:
    print("No misclassified samples to analyze or visualize.")

resnet.save_results(
    {"train_results": resnet_train_results, "test_results": resnet_test_results},
    output_path="../Results/resnet_results.json"
)
resnet.save_model(model_path="../models/resnet_classifier.pth")

# -------------------
# ViT Classifier
# -------------------
print("\nTraining ViT Classifier...")
vit = ViTClassifier(device=device)
vit_train_loader = DataLoader(
    ImageDataset(images_train, labels_train, transform=vit_transform),
    batch_size=32, shuffle=True
)
vit_val_loader = DataLoader(
    ImageDataset(images_val, labels_val, transform=vit_transform),
    batch_size=32, shuffle=False
)
vit_test_loader = DataLoader(
    ImageDataset(images_test, labels_test, transform=vit_transform),
    batch_size=32, shuffle=False
)
vit_train_results = vit.train_with_validation(vit_train_loader, vit_val_loader, epochs=20)
vit_test_results, vit_misclassified_samples = vit.evaluate_model(vit_test_loader)

if vit_misclassified_samples:
    visualize_errors(vit_misclassified_samples, images_test)
    vit_error_summary = analyze_errors(vit_misclassified_samples)
    save_error_analysis(vit_error_summary, output_path="../Results/error_analysis_vit.json")
else:
    print("No misclassified samples to analyze or visualize.")

# Save results and model
vit.save_results(
    {"train_results": vit_train_results, "test_results": vit_test_results},
    output_path="../Results/vit_results.json"
)
vit.save_model(model_path="../models/vit_classifier.pth")