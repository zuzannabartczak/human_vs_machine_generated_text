import json
import gc
import torch
import pandas as pd
import tensorflow as tf
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load the pre-trained BERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the neural network architecture
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

# Load the data from the JSONL file into a pandas DataFrame
with open('/domus/h1/zuzanna/hvm/SubtaskA/subtaskA_train_monolingual.jsonl', 'r') as f:
    df = pd.read_json(f, lines=True, orient='records')

class task_A_dataloader(Dataset):
    def __init__(self, df, text, label, model, source, id):
        self.text = df['text']
        self.label = df['label']
        self.model = df['model']
        self.source = df['source']
        self.id = df['id']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        item_label = self.label[idx]
        item_text = self.text[idx]
        item_model = self.model[idx]
        item_source = self.source[idx]
        item_id = self.id[idx]
        return item_text, item_label, item_model, item_source, item_id

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create an instance of the task_A_dataloader class for the test set
test_dataset = task_A_dataloader(test_df, 'text', 'label', 'model', 'source', 'id')
train_dataset = task_A_dataloader(train_df, 'text', 'label', 'model', 'source', 'id')

# Encode the input data using the tokenizer on the GPU
encoded_data_train = []
for text in tqdm(train_dataset.text.tolist()):
    encoded_text = tokenizer(text, max_length=512, truncation=True, padding='max_length')
    encoded_data_train.append(encoded_text)

# Encode the input data using the tokenizer on the GPU
encoded_data_test = []
for text in tqdm(test_dataset.text.tolist()):
    encoded_text = tokenizer(text, max_length=512, truncation=True, padding='max_length')
    encoded_data_test.append(encoded_text)

# Create data loaders for training and validation sets
train_input_ids_tensor = torch.tensor([data['input_ids'] for data in encoded_data_train])
train_attention_masks_tensor = torch.tensor([data['attention_mask'] for data in encoded_data_train])
train_labels_tensor = torch.tensor(train_df['label'].values)

val_input_ids_tensor = torch.tensor([data['input_ids'] for data in encoded_data_test])
val_attention_masks_tensor = torch.tensor([data['attention_mask'] for data in encoded_data_test])
val_labels_tensor = torch.tensor(test_df['label'].values)

train_loader = torch.utils.data.TensorDataset(train_input_ids_tensor, train_attention_masks_tensor, train_labels_tensor)
val_loader = torch.utils.data.TensorDataset(val_input_ids_tensor, val_attention_masks_tensor, val_labels_tensor)

# Define the optimizer, loss, and metric
optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-5)
loss = nn.CrossEntropyLoss()
metric = accuracy_score

from sklearn.metrics import accuracy_score

# Define the training loop
def train(model, loss_fn, metric, optimizer, train_loader, val_loader, epochs=10, device='cuda'):
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # Move the model to the specified device
    model.to(device)

    # Set the model to training mode
    model.train()

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0

        # Iterate over the training data
        train_loader = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{epochs}")
        for inputs, attention_masks, labels in train_loader:
            # Move the inputs, attention masks, and labels to the specified device
            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(0)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, attention_masks)
            loss = loss_fn(outputs.logits.cpu(), labels.cpu())
            preds = torch.argmax(outputs.logits.detach().cpu(), dim=1)
            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the training loss and accuracy
            train_loss += loss.item()
            train_acc += metric(preds, labels.cpu())

        # Calculate the average training loss and accuracy
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Set the model to evaluation mode
        model.eval()

        val_loss = 0.0
        val_acc = 0.0

        val_loader = tqdm(val_loader, desc=f"Validating epoch {epoch + 1}/{epochs}")
        # Disable gradient calculation
        with torch.no_grad():
            # Iterate over the validation data
            for inputs, attention_masks, labels in val_loader:
                # Move the inputs, attention masks, and labels to the specified device
                inputs = inputs.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(0)

                # Forward pass
                outputs = model(inputs, attention_masks)
                loss = loss_fn(outputs.logits.cpu(), labels.cpu())
                preds = torch.argmax(outputs.logits.detach().cpu(), dim=1)
                # Update the validation loss and accuracy
                val_loss += loss.item()
                val_acc += metric(preds, labels.cpu())

        # Calculate the average validation loss and accuracy
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # Append the training and validation metrics to the lists
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # Print the training and validation metrics for each epoch
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print()

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list

# Train the model using the training loop
train_loss_list, train_acc_list, val_loss_list, val_acc_list = train(model, loss, metric, optimizer, train_loader, val_loader)

# Extract the logits from the SequenceClassifierOutput object
train_logits = train_loss_list.logits
val_logits = val_loss_list.logits

# Convert the logits to tensors
train_logits_tensor = torch.tensor(train_logits)
val_logits_tensor = torch.tensor(val_logits)

# Print the training and validation metrics
print("Training Loss:", train_loss_tensor[-1])
print("Training Accuracy:", train_acc_tensor[-1])
print("Validation Loss:", val_loss_list[-1])
print("Validation Accuracy:", val_acc_list[-1])