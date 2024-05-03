# Denoising Diffusion Model

This repository contains a Python script-based implementation of a Denoising Diffusion Probabilistic Model (DDPM) for image generation. The project is built using PyTorch and trains a model on the Fashion-MNIST dataset.

The commit history of this repository is designed to show the logical progression of building a diffusion model, from the foundational components to a full training and sampling pipeline.

## Project Overview

The core idea of diffusion models is to:
1.  **Forward Process:** Gradually add noise to an image over a series of timesteps until it becomes pure random noise.
2.  **Reverse Process:** Train a neural network (typically a U-Net) to learn how to reverse this process, starting from noise and iteratively denoising it to generate a clean image.

### Project Structure
-   `unet.py`: Defines the U-Net architecture used as the denoising model.
-   `train.py`: The main script that handles:
    -   Loading and preparing the Fashion-MNIST dataset.
    -   Defining the diffusion schedule (beta schedule).
    -   Implementing the training loop to teach the U-Net to predict noise.
    -   Implementing the sampling loop to generate new images from noise.
-   `output/`: A directory where generated sample images will be saved.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/adityasengar/diffusion_model.git
    cd diffusion_model
    ```

2.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

Note: The first run will download the Fashion-MNIST dataset, which may take a moment.

To train the diffusion model and generate new images, simply run the `train.py` script.

```bash
python train.py
```

The script will:
1.  Download the Fashion-MNIST dataset.
2.  Train the U-Net model for a set number of epochs (this may take some time, especially without a GPU).
3.  Save the trained model weights to `diffusion_model.pth`.
4.  Use the trained model to sample a batch of new images.
5.  Save the generated images in a grid to `output/generated_samples.png`.

```