# Bio_Clinical_BERT_Multimodal_Model
A predictive model for glaucoma surgical outcomes that utilizes Bio-Clinical BERT and a multimodal neural network. The model incorporates both structured EHR data and free-text operative notes.

# PyTorch Text Classification Model with Transformer Encoder

This project contains a PyTorch implementation for classifying text with the help of transformer encoders. It utilizes Word2Vec for embeddings, and several other techniques like Positional Encoding, Dropout, and Adam optimizer.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Code Structure](#code-structure)
- [Training the Model](#training-the-model)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Authors](#authors)
- [License](#license)
- [Contact](#contact)

## Features

- Transformer Encoder Layer
- Positional Encoding
- Adam optimizer with learning rate scheduler
- Class weights for imbalanced classes
- Cross Entropy Loss function
- Pretrained Word2Vec model for embeddings

## Requirements

- PyTorch
- NumPy
- Matplotlib
- tqdm
- scikit-learn

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo.git
    ```
2. Navigate to the project directory and install requirements:
    ```bash
    cd project-directory
    pip install -r requirements.txt
    ```
3. Run the main script to train the model:
    ```bash
    python main.py
    ```

## Code Structure

- `Word2Vec.load()`: For loading pre-trained Word2Vec models.
- `PositionalEncoding`: Class for generating positional encoding.
- `Multi_TF_Class`: The main classification model class with transformer encoder.
- `train()`: Function to train the model.
- `evaluate()`: Function to evaluate the model.
- `get_accuracy()`: Function to compute accuracy during training and validation.

## Training the Model

- **Batch Size**: 16
- **Epochs**: 250
- **Learning Rate**: 4e-5
- **Weight Decay for L2 Regularization**: 1e-6
- **Class Weights**: [0.2584, 0.8678, 0.8737]
- **Learning Rate Scheduler**: Reduces learning rate by a factor of 0.5 every 100 epochs

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Acknowledgments

- Special thanks to the creators of PyTorch and scikit-learn for their well-documented libraries.
- Many thanks to all who contributed to the pre-trained Word2Vec models.
  
## Authors

- [Your Name](https://github.com/your-github-profile)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

- **Email**: your.email@example.com
- **Twitter**: [@your_username](https://twitter.com/your_username)

Feel free to contact me if you have any questions or want to contribute.
