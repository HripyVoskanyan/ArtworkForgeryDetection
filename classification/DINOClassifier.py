import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torchvision.transforms.functional import to_pil_image
import timm
from early_stopping import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# Dataset class for DINO preprocessing
class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# Dataset class for embedding-based training
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# DINO Classifier Model
class DINOClassifier(nn.Module):
    def __init__(self, embedding_dim=768):  # Default for DINO ViT-B/16
        super(DINOClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class DINOEmbeddingPipeline:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0).to(self.device)

        # Define DINO preprocessing
        self.transform = Compose([
            Resize((224, 224)),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.early_stopping = EarlyStopping(patience=5, min_delta=0.005)

    def split_dataset(self, images, labels, test_size=0.4, val_size=0.5):
        images_train, images_temp, labels_train, labels_temp = train_test_split(
            images, labels, test_size=test_size, stratify=labels, random_state=42
        )
        images_val, images_test, labels_val, labels_test = train_test_split(
            images_temp, labels_temp, test_size=val_size, stratify=labels_temp, random_state=42
        )
        return images_train, labels_train, images_val, labels_val, images_test, labels_test

    def extract_embeddings(self, dataloader):
        embeddings = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for images, batch_labels in dataloader:
                images = images.to(self.device)  # Images are already tensors
                embedding = self.model(images).squeeze(0)  # Extract embedding
                embeddings.append(embedding.cpu())
                labels.extend(batch_labels)
        return torch.stack(embeddings), torch.tensor(labels)

    def train_classifier(self, model, train_loader, val_loader, criterion, optimizer, epochs=10):
        results = {"epochs": []}
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for embeddings, labels in train_loader:
                embeddings, labels = embeddings.to(self.device), labels.to(self.device).float()

                optimizer.zero_grad()
                outputs = model(embeddings).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            y_true, y_pred = [], []
            with torch.no_grad():
                for embeddings, labels in val_loader:
                    embeddings, labels = embeddings.to(self.device), labels.to(self.device).float()
                    outputs = model(embeddings).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predictions.cpu().numpy())

            val_accuracy = accuracy_score(y_true, y_pred)
            val_recall = recall_score(y_true, y_pred)
            val_f1 = f1_score(y_true, y_pred)

            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "val_loss": val_loss / len(val_loader),
                "val_accuracy": val_accuracy,
                "val_recall": val_recall,
                "val_f1": val_f1,
            }
            results["epochs"].append(epoch_result)
            print(epoch_result)

            # Early stopping check
            if self.early_stopping.should_stop(val_loss, epoch):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return results

    def evaluate_classifier(self, model, test_loader):
        model.eval()
        y_true, y_pred, y_probs = [], [], []
        misclassified_samples = []
        with torch.no_grad():
            for idx, (embeddings, labels) in enumerate(test_loader):
                embeddings, labels = embeddings.to(self.device), labels.to(self.device).float()
                outputs = model(embeddings).squeeze()
                predictions = (outputs > 0.5).float()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
                y_probs.extend(outputs.cpu().numpy())

                # Collect misclassified samples
                for i, (label, prediction, confidence) in enumerate(zip(labels, predictions, outputs)):
                    if label != prediction:  # Misclassified
                        misclassified_samples.append({
                            "image_index": idx * test_loader.batch_size + i,
                            "true_label": label.item(),
                            "predicted_label": prediction.item(),
                            "confidence": confidence.item()
                        })

        test_accuracy = accuracy_score(y_true, y_pred)
        test_recall = recall_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred)
        results = {
            "test_accuracy": test_accuracy,
            "test_recall": test_recall,
            "test_f1": test_f1,
        }
        print(results)
        return results, misclassified_samples

    def save_results(self, results, output_path="../Results/dino_results.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    def save_model(self, model, model_path="../models/dino_classifier.pth"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

