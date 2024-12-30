# Fashion MNIST Autoencoder

This repository contains an implementation of an autoencoder trained on the Fashion MNIST dataset. The autoencoder compresses images into a 2D latent space for visualization and reconstructs images from their latent representations.

## Features
- **Data Preprocessing:** Normalizes and pads images to fit the model's input requirements.
- **Encoder-Decoder Architecture:** 
  - The encoder compresses images into a 2D latent space.
  - The decoder reconstructs images from latent embeddings.
- **Visualization:**
  - Embeddings in the latent space colored by clothing type.
  - Reconstructed images compared to original images.
  - Generation of new images by sampling points in the latent space.
- **Model Saving:** Trained models and weights are saved for reuse.

## Requirements
Install the required Python packages using:
```bash
pip install -r requirements.txt
