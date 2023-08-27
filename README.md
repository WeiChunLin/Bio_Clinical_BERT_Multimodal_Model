# Bio-Clinical BERT Multimodal Model for Glaucoma Surgery Outcome Prediction

This repository provides the codebase for the multi-modal predictive model for glaucoma surgical outcomes. We utilize Bio-Clinical BERT to extract information from operative notes and combine it with static features. The code is written in Python and uses PyTorch for the deep learning components.

## Table of Contents
- [Data Structure](#data-structure)
- [Functionalities](#functionalities)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)

## Data Structure

The dataset should include:

- A target column named `output`, which contains three classes for prediction.
- A feature column named `clean-up operative notes` that includes the operative notes.
- 75 columns of structured EHR data.

## Functionalities

This codebase is designed to:

1. Import an Excel file into a Pandas DataFrame.
2. Split the feature and target variables into training, validation, and test sets (70%/10%/20%).
3. Initialize a ClinicalBERT tokenizer for text processing.
4. Tokenize the clean-up operative notes and create data loaders for training, validation, and testing.
5. Utilize an input dimension of [512, 512, 75, 1], where:
   - The first two dimensions are the encoded input IDs and attention masks from Bio-Clinical BERT.
   - The third dimension contains structured EHR (Electronic Health Records) data.
   - The fourth dimension holds the outcome labels.
6. Set up a PyTorch model class that combines a pre-trained Bio_ClinicalBERT model with static data.
7. Initialize and configure the Multi_BERT model and move it to the specified device (CPU or GPU).
8. Set three functions: get_accuracy, train, and evaluate.
9. Start to train the PyTorch model using a specified set of parameters, optimizer, and loss function.
10. Evaluate the model on a test set to generate predictions with ROCs, P-R curves, and classification reports.

## Getting Started

### Prerequisites

Ensure that you have:
- Python 
- PyTorch
- Transformers library from Hugging Face
- Other requirements 

### Installation

To install the necessary packages, run the following command:

```bash
pip install torch pandas transformers
```
## Code structure
- `Multi_TF_Class`: The main classification model class with transformer encoder.
- `train()`: Function to train the model.
- `evaluate()`: Function to evaluate the model.
- `get_accuracy()`: Function to compute accuracy during training and validation.
  
## Model Architecture

Our model leverages a pre-trained ClinicalBERT and adds custom layers for dimensionality reduction and combining with static data for classification tasks. The architecture is defined in the `Multi_BERT` class.

## Training

The model is trained using a defined set of hyperparameters, a specified loss function, and an optimizer. 

## Evaluation
The model is evaluated using a separate test dataset. Evaluation metrics include AUC (Area Under the Curve), Precision-Recall Curve, and Classification reports.

## License
This project is licensed under the MIT License.
