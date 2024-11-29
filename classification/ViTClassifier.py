import os
import json
import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from early_stopping import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
class ViTClassifier:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = vit_b_16(pretrained=True)

        # Modify the classifier head for binary classification
        self.model.heads = nn.Sequential(
            nn.Linear(self.model.heads.head.in_features, 1),  # Single output
            nn.Sigmoid()  # Output probabilities
        )
        self.model = self.model.to(self.device)
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    def train_with_validation(self, train_loader, val_loader, epochs=10):
        self.model.train()
        results = {"epochs": []}

        for epoch in range(epochs):
            train_loss = 0.0

            # Training phase
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Validation phase
            val_loss = 0.0
            y_true, y_pred = [], []
            self.model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device).float()
                    outputs = self.model(images).squeeze()
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predictions.cpu().numpy())

            # Metrics
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

    def evaluate_model(self, test_loader, save_errors_path="../Results/misclassified_samples_vit.json"):
        y_true, y_pred, y_probs = [], [], []
        misclassified_samples = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device).float()
                outputs = self.model(images).squeeze()
                predictions = (outputs > 0.5).float()
                probabilities = outputs.cpu().numpy()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
                y_probs.extend(probabilities)

                for i in range(len(labels)):
                    if predictions[i] != labels[i]:
                        misclassified_samples.append({
                            "image_index": i,
                            "true_label": int(labels[i].cpu().numpy()),
                            "predicted_label": int(predictions[i].cpu().numpy()),
                            "confidence": float(probabilities[i])  # Convert to native Python float
                        })

        # Save misclassified samples
        os.makedirs(os.path.dirname(save_errors_path), exist_ok=True)
        with open(save_errors_path, "w") as f:
            json.dump(misclassified_samples, f, indent=4)

        # Metrics
        test_accuracy = accuracy_score(y_true, y_pred)
        test_recall = recall_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred)

        results = {
            "test_accuracy": test_accuracy,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "misclassified_count": len(misclassified_samples),
        }
        print(results)

        return results, misclassified_samples

    def save_results(self, results, output_path="../Results/vit_results.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    def save_model(self, model_path="../models/vit_artwork_classifier.pth"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
