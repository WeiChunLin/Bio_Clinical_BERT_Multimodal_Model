#Import packages
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn import metrics
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC
import time
import copy
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.optim.lr_scheduler as lr_scheduler

"""
Script for Data Preprocessing and Tokenizer Initialization

This script reads an Excel file, splits the dataset into training, validation, and test sets, and initializes a ClinicalBERT tokenizer.

Data Structure:
    - The target variable (label) for prediction is stored in a column named 'output' with 3 classes.
    - The first feature column is clean-up operative notes.

Functionalities:
    1. Reads the Excel file into a Pandas DataFrame.
    2. Splits the DataFrame into feature and label sets.
    3. Further splits these into training, validation, and test sets.
    4. Outputs the shapes of these datasets for verification.
    5. Separates dynamic features from static features.
    6. Initializes a ClinicalBERT tokenizer for further processing.
    7. Tokenize the notes and create training, validation, and testing dataloaders.
    8. Input dimensions are [512, 512, 75, 1].
    9. First two dimensions represent the encoded input IDs and attention masks from Bio-Clinical BERT.
    10. The third dimension represents the structured EHR data.
    11. The fourth dimension represents the outcome labels.
"""

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel('glaucoma_surgery_dataset.xlsx', sheet_name='Sheet1')

# Isolate the target variable (label) which we want to predict
labels = df['output']
# Remove the target variable from the feature set; axis=1 means we drop a column not a row
features = df.drop('output', axis=1)

# Split the data into a training set and a temporary validation/test set
# We're using 30% of the data for the temporary validation/test set, stratified by the label
train_features, val_test_features, train_labels, val_test_labels = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

# Further split the temporary validation/test set into validation and test sets
# 2/3 of the data goes to the test set, stratified by the label
val_features, test_features, val_labels, test_labels = train_test_split(val_test_features, val_test_labels, test_size=(2/3), random_state=42, stratify=val_test_labels)

# Display shapes to ensure everything is as expected
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Val Features Shape:', val_features.shape)
print('Val Labels Shape:', val_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Select static features by dropping the first column (operative note) from the train, validation and test feature sets
train_static = train_features.iloc[:, 1:]
val_static = val_features.iloc[:, 1:]
test_static = test_features.iloc[:, 1:]

# Initialize the ClinicalBERT tokenizer
tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

#Tokenize the operative notes using BertTokenizer from the Transformers library.

#Generate the dataset and dataloader using TensorDataset and DataLoader from PyTorch.

'''
Sets random seeds to ensure reproducibility across different runs.
'''
# Set a random seed for reproducibility across runs
random_seed = 101  # Or any other favorite number you prefer
# Set the random seed for PyTorch (both CPU and CUDA)
torch.manual_seed(random_seed)
# Set the random seed for CUDA (GPU)
torch.cuda.manual_seed(random_seed)
# Make CuDNN deterministic to ensure reproducibility
# (may slow down the computations)
torch.backends.cudnn.deterministic = True
# Enable CuDNN benchmarking for potentially better performance
# (this should be enabled only if the input sizes do not vary)
torch.backends.cudnn.benchmark = True
# Set the random seed for NumPy
np.random.seed(random_seed)
# Set the random seed for Python's built-in random module
random.seed(random_seed)

"""
Start to train the PyTorch model using a specified set of parameters, optimizer, and loss function.
"""
   
# Check if CUDA is available and set the device to GPU if possible, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training parameters
epochs = 200  # Number of training epochs
LEARNING_RATE = 4e-5  # Learning rate for the optimizer
WEIGHT_DECAY = 1e-5  # Weight decay for L2 regularization
class_weights = torch.tensor([0.2584, 0.8678, 0.8737]).to(device)  # Class weights for handling class imbalance

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss(weight=class_weights)  # Cross-entropy loss function with class weights
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # Adam optimizer with learning rate and weight decay

# Convert validation labels to a list
# This is done to feed the labels into the evaluation function
y_val = val_labels.tolist()

# Train the model
# Calls the 'train' function and stores returned training and validation losses
train_losses, val_losses = train(model, optimizer, loss_fn, train_dataloader, val_dataloader, y_val, epochs, batchSize=16)

# Plotting the losses
# Plots both training and validation losses over epochs
plt.plot(train_losses, label='train loss')  # Plot training losses
plt.plot(val_losses, label='test loss')  # Plot validation/test losses
plt.legend()  # Add legend to the plot
plt.show()  # Display the plot

# Saving the trained model weights to disk
PATH = 'Bio-Clinical-BERT-Mulimodal.pth'
torch.save(model.state_dict(), PATH)


"""
A PyTorch model class that combines a pre-trained Bio_ClinicalBERT model with static data and additional
fully connected layers for classification tasks.

Parameters:
- bert_model_name (str): The name of the pre-trained BERT model to use. Default is 'emilyalsentzer/Bio_ClinicalBERT'.
- classifier_dropout (float): Dropout rate for the fully connected layers. Default is 0.5.
- n_node_layer1 (int): Number of nodes in the first fully connected layer. Default is 256.
- n_node_layer2 (int): Number of nodes in the second fully connected layer. Default is 48.
- static_size (int): The dimension of the static input features. Default is 75.

Methods:
- forward(input_ids, attention_mask, x_static): Forward pass for the model.
"""

class Multi_BERT(nn.Module):

    def __init__(
        self,
        bert_model_name='emilyalsentzer/Bio_ClinicalBERT',  # Pre-defined model name
        classifier_dropout=0.5,
        n_node_layer1=256,
        n_node_layer2=48,
        static_size=75):

        super(Multi_BERT, self).__init__()

        # Load the Bio_ClinicalBERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Extract the hidden size from BERT's configurations (usually 768 for base models)
        d_model = self.bert.config.hidden_size

        # Linear layer to reduce the dimensionality of BERT's output
        self.bert_output_reducer = nn.Linear(d_model, 50)

        # Static data dimensions
        self.static_size = static_size

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=classifier_dropout)

        # Batch normalization layers to stabilize learning
        self.batchnorm1 = nn.BatchNorm1d(n_node_layer1, momentum=0.1)
        self.batchnorm2 = nn.BatchNorm1d(n_node_layer2, momentum=0.1)
        
        # Define the fully connected layers
        self.linear1 = nn.Linear(50 + static_size, n_node_layer1)  # Combine BERT and static data
        self.relu = nn.ReLU()  # Activation function
        self.linear2 = nn.Linear(n_node_layer1, n_node_layer2)
        self.classifier = nn.Linear(n_node_layer2, 3)  # Output layer

    def forward(self, input_ids, attention_mask, x_static):
        """
        Forward pass for the model.
        """

        # Pass input through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the [CLS] token's representation (used for classification tasks)
        bert_output = outputs[0][:, 0, :]

        # Reduce the dimensionality of the BERT output
        bert_output = self.bert_output_reducer(bert_output)
        
        # Concatenate BERT output and static data for the fully connected layers
        inputs = torch.cat([bert_output, x_static], dim=1)
        
        # Pass through the first fully connected layer and apply ReLU and BatchNorm
        out = self.relu(self.linear1(inputs))
        out = self.batchnorm1(out)
        out = self.dropout(out)

        # Pass through the second fully connected layer and apply ReLU and BatchNorm
        out = self.relu(self.linear2(out))
        out = self.batchnorm2(out)
        out = self.dropout(out)

        # Pass through the output layer
        out = self.classifier(out)

        return out
    
"""
Initialize and configure the Multi_BERT model, and move it to the specified device (CPU or GPU).
Model (Multi_BERT): The initialized and configured Multi_BERT model moved to the specified device.
"""

# Initialize the Multi_BERT model with specified hyperparameters
model = Multi_BERT(
    bert_model_name='emilyalsentzer/Bio_ClinicalBERT',  # Pre-trained model to use
    classifier_dropout=0.5,  # Dropout rate
    n_node_layer1=256,  # Number of nodes in the first fully connected layer
    n_node_layer2=48,   # Number of nodes in the second fully connected layer
    static_size=75,     # Size of the static features
)

# Make all BERT model parameters trainable
for param in model.bert.parameters():
    param.requires_grad = True

# Move the model to the appropriate processing device (GPU or CPU)
model.to(device)

def get_accuracy(out, actual_labels, batchSize):
    '''
    Computes the accuracy of a model's predictions for a given batch.
    
    Parameters:
    - out (Tensor): The log probabilities or logits returned by the model.
    - actual_labels (Tensor): The actual labels for the batch.
    - batchSize (int): The size of the batch.
    
    Returns:
    float: The accuracy for the batch.
    '''
    # Get the predicted labels from the maximum value of log probabilities
    predictions = out.max(dim=1)[1]
    # Count the number of correct predictions
    correct = (predictions == actual_labels).sum().item()
    # Compute the accuracy for the batch
    accuracy = correct / batchSize
    
    return accuracy

def train(model, optimizer, loss_fn, train_dataloader, val_dataloader, y_val, epochs=20, batchSize=16):
    '''
    Train a PyTorch model using the given parameters and dataloaders.
    
    Parameters:
    - model (nn.Module): The PyTorch model to train.
    - optimizer (torch.optim.Optimizer): The optimizer for training.
    - loss_fn (callable): The loss function.
    - train_dataloader (DataLoader): The DataLoader for the training data.
    - val_dataloader (DataLoader): The DataLoader for the validation data.
    - y_val (array-like): The true labels for the validation data.
    - epochs (int, optional): The number of training epochs. Default is 20.
    - batchSize (int, optional): The size of each batch. Default is 16.
    
    Returns:
    tuple: The training and validation losses for each epoch from validation.
    '''
    
    # Initialize device to GPU if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize metrics to track best validation loss, accuracy, and AUC
    best_val_loss = 2
    best_accuracy = 0
    best_AUC = 0
    best_p = []  # For storing best probability scores
    
    # Initialize arrays to store training and validation losses for each epoch
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    # Print the header for the training log
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val AUC':^9} | {'Val F1':^9} |{'Elapsed':^9}")
    print("-"*60)

    # Loop through each epoch
    for epoch_i in tqdm(range(epochs)):
        # Record time at start of epoch
        t0_epoch = time.time()
        
        # Initialize metrics for the current epoch
        total_loss = 0
        epoc_acc = 0
        
        # Set the model to training mode
        model.train()

        # Loop through each batch of data in the training dataloader
        for step, batch in enumerate(train_dataloader):
            # Load the current batch
            b_input_ids, b_attention_mask, b_input_tbl, b_labels = batch
            b_input_ids, b_attention_mask, b_input_tbl, b_labels = b_input_ids.to(device), b_attention_mask.to(device), b_input_tbl.to(device), b_labels.long().to(device)
            
            # Clear the gradients
            model.zero_grad()

            # Forward pass
            logits = model(b_input_ids, b_attention_mask, b_input_tbl)
            logits = logits.float().to(device)
            
            # Compute loss
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()
            
            # Compute accuracy
            epoc_acc += get_accuracy(logits, b_labels, batchSize)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        # Compute average loss and accuracy over the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = epoc_acc / len(train_dataloader)
        train_losses[epoch_i] = avg_train_loss

        # Validate the model if a validation dataloader is provided
        if val_dataloader is not None:
            val_loss, val_accuracy, val_AUC, val_f1, p = evaluate(model, val_dataloader, y_val, batchSize=16)
            
            val_losses[epoch_i] = val_loss

            # Update best metrics if current epoch's metrics are better
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
            if val_AUC > best_AUC:
                best_AUC = val_AUC
                best_p = p
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts_BERTv1 = copy.deepcopy(model.state_dict())
                
            # Compute elapsed time for the epoch
            time_elapsed = time.time() - t0_epoch
            
            # Log training and validation metrics
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {avg_train_acc:^9.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {val_AUC:^9.4f} | {val_f1:^9.4f} | {time_elapsed:^9.2f}")
            
    # Print final best metrics
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%.")
    print(f"Training complete! Best AUC: {best_AUC:.4f}.")
    
    # Return training and validation losses, and the best probability scores
    return train_losses, val_losses

def evaluate(model, val_dataloader, y_val, batchSize=16):
    """
    Evaluates the model on the validation dataset.
    
    Parameters:
    - model (torch.nn.Module): The model to be evaluated.
    - val_dataloader (DataLoader): DataLoader for the validation dataset.
    - y_val (array): Array of validation labels.
    - batchSize (int, optional): Batch size. Default is 16.

    Returns:
    - val_loss (float): Average validation loss.
    - val_accuracy (float): Average validation accuracy.
    - val_AUC (float): Area under the ROC curve for the validation set.
    - val_f1 (float): F1 score for the validation set.
    - p (array): Probabilities of each class for each sample in the validation set.
    """
    
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store various metrics
    val_accuracy = []
    val_loss = []
    outputs_list = []
    y_pred_list = []
    probs_list = []

    # Initialize tensor placeholders for predictions and true labels
    preds_list = torch.tensor([], dtype=torch.long, device=device)
    labels_list = torch.tensor([], dtype=torch.long, device=device)

    # Loop through each batch of data in the validation dataloader
    for batch in val_dataloader:
        # Load the current batch
        b_input_ids, b_attention_mask, b_input_tbl, b_labels = batch
        b_input_ids, b_attention_mask, b_input_tbl, b_labels = b_input_ids.to(device), b_attention_mask.to(device), b_input_tbl.to(device), b_labels.long().to(device)

        # Forward pass with no gradient calculation
        with torch.no_grad():
            logits = model(b_input_ids, b_attention_mask, b_input_tbl)
            logits = logits.float().to(device)
        
        # Compute softmax probabilities
        y_val_probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Append logits and probabilities to respective lists
        outputs_list.append(logits)
        probs_list.append(y_val_probs)
        
        # Extract predicted labels (class with max logit)
        predictions = logits.max(dim=1)[1]
        
        # Concatenate predictions and true labels for this batch to existing list
        preds_list = torch.cat([preds_list, predictions])
        labels_list = torch.cat([labels_list, b_labels])

        # Compute loss for this batch and store
        loss = loss_fn(logits, b_labels).to(device)
        val_loss.append(loss.item())

        # Compute accuracy for this batch and store
        val_accuracy.append(get_accuracy(logits, b_labels, batchSize))

    # Compute mean loss and accuracy for entire validation set
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    # Convert lists to NumPy arrays for evaluation
    p = torch.cat(probs_list).detach().cpu().numpy()
    preds_list = preds_list.cpu().numpy()
    labels_list = labels_list.cpu().numpy()

    # Compute F1 score and AUC
    val_f1 = f1_score(labels_list, preds_list, average='weighted')
    val_AUC = roc_auc_score(y_val, p, multi_class='ovr')

    # Return all evaluation metrics
    return val_loss, val_accuracy, val_AUC, val_f1, p

"""
This following code performs the following operations:

1. Load the pre-trained model from the specified file.
2. Evaluate the model on a test set to generate predictions.
3. Compute and plot Receiver Operating Characteristic (ROC) curves for each class and their macro-average.
4. Compute and plot Precision-Recall (P-R) curves for each class and their macro-average.
5. Print out the classification report and the confusion matrix for model evaluation.

Outputs:
- Plots of ROC and P-R curves.
- Printed classification report and confusion matrix.
"""

# Load the best model
model.load_state_dict(torch.load('Bio-Clinical-BERT-Mulimodal.pth'))

# Get predictions for test set
model.eval()
test_probabilities = []
test_true_labels = []

for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_input_tbl = batch[2].to(device)
    b_labels = batch[3].to(device)
    
    with torch.no_grad():
        logits = model(b_input_ids, b_input_mask, b_input_tbl)
        logits = logits.float().to(device)
    
    #logits = outputs[0]
    probs = torch.nn.functional.softmax(logits, dim=1)
    test_probabilities.extend(probs.detach().cpu().numpy())
    test_true_labels.extend(b_labels.detach().cpu().numpy())

test_probabilities = np.array(test_probabilities)
test_true_labels = np.array(test_true_labels)

# Compute macro-average ROC curve and ROC area
fpr = dict()
tpr = dict()
roc_auc = dict()
all_fpr = np.linspace(0, 1, 100)

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(test_true_labels == i, test_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.4f})'.format(i, roc_auc[i]))

# Compute macro-average ROC curve and ROC area
mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= 3
mean_auc = auc(all_fpr, mean_tpr)
plt.plot(all_fpr, mean_tpr, color='b', linestyle='--', lw=2, label='Macro-average ROC (area = {0:0.4f})'.format(mean_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.show()

# Compute macro-average P-R curve and P-R area
precision = dict()
recall = dict()
average_precision = dict()

for i in range(3):
    precision[i], recall[i], _ = precision_recall_curve(test_true_labels == i, test_probabilities[:, i])
    average_precision[i] = average_precision_score(test_true_labels == i, test_probabilities[:, i])
    plt.step(recall[i], precision[i], lw=2, where='post', label='P-R curve of class {0} (area = {1:0.4f})'.format(i, average_precision[i]))

# Macro-average P-R curve and P-R area
mean_precision = sum(average_precision.values()) / 3
plt.plot([0, 1], [mean_precision, mean_precision], linestyle='--', lw=2, color='b', label='Macro-average P-R (area = {0:0.4f})'.format(mean_precision))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='upper right')
plt.title('Precision-Recall (P-R) Curves')
plt.show()

# Print classification report and confusion matrix
predicted_classes = np.argmax(test_probabilities, axis=1)
print(classification_report(test_true_labels, predicted_classes, target_names=['Success', 'Low IOP', 'High IOP']))
print("\nConfusion Matrix:\n", confusion_matrix(test_true_labels, predicted_classes))
