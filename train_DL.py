


import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, classification_report


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    early_stopping,
    scheduler,
    device
):
    """
    Train deep learning model for binary classification
    """

    train_loss_list, val_loss_list = [], []
    train_auc_list, val_auc_list = [], []

    for epoch in range(epochs):
        # -------- Training --------
        model.train()
        epoch_loss = 0
        y_true, y_score = [], []

        for data, label in train_loader:
            data = data.float().to(device).transpose(1, 2)
            label = label.float().to(device).view(-1, 1)

            optimizer.zero_grad()
            logits = model(data)
            probs = torch.sigmoid(logits)

            loss = criterion(probs, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            y_true.extend(label.cpu().numpy())
            y_score.extend(probs.detach().cpu().numpy())

        fpr, tpr, _ = roc_curve(y_true, y_score)
        train_auc = auc(fpr, tpr)

        train_loss_list.append(epoch_loss / len(train_loader))
        train_auc_list.append(train_auc)

        # -------- Validation --------
        model.eval()
        val_loss = 0
        y_true, y_score, y_pred = [], [], []

        with torch.no_grad():
            for data, label in val_loader:
                data = data.float().to(device).transpose(1, 2)
                label = label.float().to(device).view(-1, 1)

                probs = torch.sigmoid(model(data))
                loss = criterion(probs, label)

                val_loss += loss.item()
                y_score.extend(probs.cpu().numpy())
                y_true.extend(label.cpu().numpy())
                y_pred.extend((probs > 0.5).int().cpu().numpy())

        fpr, tpr, _ = roc_curve(y_true, y_score)
        val_auc = auc(fpr, tpr)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss_list[-1]:.4f}, Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f}, Val AUC: {val_auc:.4f}"
        )

        print(classification_report(y_true, y_pred))

        val_loss_list.append(val_loss / len(val_loader))
        val_auc_list.append(val_auc)

        early_stopping(-val_auc, model)
        scheduler.step()

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return train_loss_list, val_loss_list, train_auc_list, val_auc_list























