import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torchvision.transforms.functional import to_pil_image
import clip
from early_stopping import EarlyStopping
import warnings

warnings.filterwarnings("ignore")


# Dataset class for image preprocessing
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



# Model for classification
class CLIPViTClassifier(nn.Module):
    def __init__(self, embedding_dim=512, dropout_rate=0.5):  # Default for CLIP ViT-B/32
        super(CLIPViTClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),  # First hidden layer
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),  # Second hidden layer
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),  # Output layer for binary classification
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.fc(x)


# Task-specific loss functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = torch.nn.functional.pairwise_distance(anchor, positive)
        negative_distance = torch.nn.functional.pairwise_distance(anchor, negative)
        loss = torch.clamp(positive_distance - negative_distance + self.margin, min=0.0)
        return loss.mean()


# Pipeline for CLIP + ViT classification
class ViTCLIPPipeline:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self.clip_transform = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                      std=(0.26862954, 0.26130258, 0.27577711))
        ])

        self.early_stopping = EarlyStopping(patience=5, min_delta=0.005)

    def create_pairs(self, embeddings, labels):
        """
        Create pairs of embeddings and corresponding similarity labels (1 for similar, 0 for dissimilar).
        Args:
            embeddings (Tensor): Embeddings for a batch of data.
            labels (Tensor): Ground truth labels for the batch.
        Returns:
            Tuple: (output1, output2, pair_labels) where:
                - output1, output2 are paired embeddings.
                - pair_labels are binary labels indicating similarity (1) or dissimilarity (0).
        """
        output1, output2, pair_labels = [], [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                output1.append(embeddings[i])
                output2.append(embeddings[j])
                pair_labels.append(1 if labels[i] == labels[j] else 0)
        return torch.stack(output1), torch.stack(output2), torch.tensor(pair_labels).to(self.device)



    def split_dataset(self, images, labels, test_size=0.4, val_size=0.5):
        images_train, images_temp, labels_train, labels_temp = train_test_split(
            images, labels, test_size=test_size, stratify=labels, random_state=42
        )
        images_val, images_test, labels_val, labels_test = train_test_split(
            images_temp, labels_temp, test_size=val_size, stratify=labels_temp, random_state=42
        )
        return images_train, labels_train, images_val, labels_val, images_test, labels_test

    def extract_clip_embeddings(self, dataloader):
        embeddings = []
        labels = []
        self.clip_model.eval()
        with torch.no_grad():
            for images, batch_labels in dataloader:
                images = images.to(self.device)
                embedding = self.clip_model.encode_image(images).squeeze(0)  # Extract embedding
                embeddings.append(embedding.cpu())
                labels.extend(batch_labels)
        return torch.stack(embeddings), torch.tensor(labels)

    def train_classifier(self, model, train_loader, val_loader, criterion, optimizer, epochs=10):
        results = {"epochs": []}
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            y_true_train, y_pred_train = [], []

            for embeddings, labels in train_loader:
                embeddings, labels = embeddings.to(self.device), labels.to(self.device).float()

                if isinstance(criterion, ContrastiveLoss):
                    # Generate pairs for Contrastive Loss
                    output1, output2, pair_labels = self.create_pairs(embeddings, labels)
                    optimizer.zero_grad()
                    outputs1, outputs2 = model(output1), model(output2)
                    loss = criterion(outputs1, outputs2, pair_labels)
                else:
                    # Standard forward pass for other losses
                    optimizer.zero_grad()
                    outputs = model(embeddings).squeeze()
                    loss = criterion(outputs, labels)

                    # Collect predictions for metrics (non-pair-based losses)
                    y_true_train.extend(labels.cpu().numpy())
                    y_pred_train.extend((outputs > 0.5).float().cpu().numpy())

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Compute training metrics (for non-pair-based losses)
            if not isinstance(criterion, ContrastiveLoss):
                train_accuracy = accuracy_score(y_true_train, y_pred_train)
                train_recall = recall_score(y_true_train, y_pred_train)
                train_f1 = f1_score(y_true_train, y_pred_train)
            else:
                train_accuracy = train_recall = train_f1 = None

            # Validation phase
            model.eval()
            val_loss = 0.0
            y_true_val, y_pred_val = [], []

            with torch.no_grad():
                for embeddings, labels in val_loader:
                    embeddings, labels = embeddings.to(self.device), labels.to(self.device).float()

                    if isinstance(criterion, ContrastiveLoss):
                        # Generate pairs for Contrastive Loss
                        output1, output2, pair_labels = self.create_pairs(embeddings, labels)
                        outputs1, outputs2 = model(output1), model(output2)
                        loss = criterion(outputs1, outputs2, pair_labels)

                        # For metrics with pairs (optional)
                        distances = torch.nn.functional.pairwise_distance(outputs1, outputs2)
                        predictions = (distances < 0.5).float()  # Example threshold
                        y_true_val.extend(pair_labels.cpu().numpy())
                        y_pred_val.extend(predictions.cpu().numpy())
                    else:
                        # Standard validation pass
                        outputs = model(embeddings).squeeze()
                        loss = criterion(outputs, labels)

                        # Collect predictions for metrics
                        y_true_val.extend(labels.cpu().numpy())
                        y_pred_val.extend((outputs > 0.5).float().cpu().numpy())

                    val_loss += loss.item()

            # Compute validation metrics
            val_accuracy = accuracy_score(y_true_val, y_pred_val)
            val_recall = recall_score(y_true_val, y_pred_val)
            val_f1 = f1_score(y_true_val, y_pred_val)

            # Save results for this epoch
            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "train_accuracy": train_accuracy,
                "train_recall": train_recall,
                "train_f1": train_f1,
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
        misclassified_samples = []  # To store misclassified samples

        with torch.no_grad():
            for batch_idx, (embeddings, labels) in enumerate(test_loader):
                embeddings, labels = embeddings.to(self.device), labels.to(self.device).float()
                outputs = model(embeddings).squeeze()
                predictions = (outputs > 0.5).float()

                # Extend true, predicted, and probability lists
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
                y_probs.extend(outputs.cpu().numpy())

                # Collect misclassified samples with detailed information
                for sample_idx, (embedding, label, prediction, confidence) in enumerate(
                        zip(embeddings.cpu(), labels.cpu(), predictions.cpu(), outputs.cpu())):
                    if label != prediction:  # Misclassified
                        global_index = batch_idx * test_loader.batch_size + sample_idx
                        misclassified_samples.append({
                            "image_index": global_index,  # Global index in the dataset
                            "embedding": embedding.numpy().tolist(),
                            "true_label": label.item(),
                            "predicted_label": prediction.item(),
                            "confidence": confidence.item(),
                        })

        # Compute metrics
        test_accuracy = accuracy_score(y_true, y_pred)
        test_recall = recall_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred)
        results = {
            "test_accuracy": test_accuracy,
            "test_recall": test_recall,
            "test_f1": test_f1,
        }

        print("Evaluation Results:", results)
        print(f"Number of Misclassified Samples: {len(misclassified_samples)}")
        return results, misclassified_samples

    def save_results(self, results, output_path="../Results/vit_clip_results.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    def save_model(self, model, model_path="../models/vit_and_clip_classifier.pth"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")
