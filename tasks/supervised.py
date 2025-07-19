import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import f1_score






def supervised_evaluation(model, train_dataset, valid_dataset, feature_dim, num_epochs, batch_size, config):


    train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle = True,
                num_workers=config.NUM_WORKERS,
                drop_last = True,
            )
            
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle = False,
        num_workers=config.NUM_WORKERS,
        drop_last = True,
    )
                  
    classifier = nn.Linear(feature_dim, len(config.class_dict))  # Add linear classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)  # optimizer

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    # Training and validation loop
    num_epochs = num_epochs
    for epoch in tqdm(range(num_epochs)):
        # Training phase
        classifier.train()  # Set the model to training mode
        train_running_loss = 0.0
        train_correct_predictions = 0
        train_total_samples = 0

        all_preds = []
        all_labels = []

        for time_series, labels in (train_loader):
            time_series = time_series.to(device)
            labels = labels.to(device)

            # Forward pass
            features = model.encode(time_series)
            y_hat = classifier(features)
            
            # Flatten y_hat to have dimensions [batch_size * sequence_length, num_classes]
            y_hat_flat = y_hat.reshape(-1, y_hat.size(-1))

            # Reshape y to have dimensions [batch_size * sequence_length]
            labels_flat = labels.view(-1)

            # Compute training loss
            train_loss = criterion(y_hat_flat, labels_flat)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update training statistics
            train_running_loss += train_loss.item() * time_series.size(0)

            _, predicted = torch.max(y_hat_flat, 1)
            train_correct_predictions += (predicted == labels_flat).sum().item()

            #Store the labels for future computation of F1-score
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())

            train_total_samples += labels_flat.size(0)

        # Calculate average training loss and accuracy for the epoch
        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        train_epoch_accuracy = 100*train_correct_predictions / train_total_samples

        f1 = f1_score(all_labels, all_preds,average='weighted')
        
    # Validation phase
    classifier.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0

    with torch.no_grad():
        val_preds = []
        val_labels = []
        for time_series, labels in (valid_loader):
            time_series = time_series.to(device)
            labels = labels.to(device)

            # Forward pass
            features = model.encode(time_series)
            y_hat = classifier(features)

            # Flatten y_hat to have dimensions [batch_size * sequence_length, num_classes]
            y_hat_flat = y_hat.reshape(-1, y_hat.size(-1))

            # Reshape y to have dimensions [batch_size * sequence_length]
            labels_flat = labels.view(-1)

            # Compute validation loss
            val_loss = criterion(y_hat_flat, labels_flat)

            # Update validation statistics
            val_running_loss += val_loss.item() * time_series.size(0)

            _, predicted = torch.max(y_hat_flat, 1)
            val_correct_predictions += (predicted == labels_flat).sum().item()
            val_total_samples += labels_flat.size(0)

            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels_flat.cpu().numpy())

    # Calculate average validation loss and accuracy for the epoch
    val_epoch_loss = val_running_loss / len(valid_loader.dataset)
    val_epoch_accuracy = 100*val_correct_predictions / val_total_samples

    # Precision and recall using sklearn
    precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)

    f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
    
    
    return {f"Val Accuracy: {val_epoch_accuracy:.2f}%, F1-score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}"}