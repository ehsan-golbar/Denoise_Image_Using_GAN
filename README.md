# Denoise_Image_Using_GAN

This project aims to remove noise from images using a Generative Adversarial Network (GAN). The primary goal is to train a GAN to reconstruct clean images from noisy inputs and evaluate the performance using various metrics.

## Project Overview
The project involves the following key steps:
1. **Data Loading and Preprocessing**:
    - The dataset used is **fer2013.csv**, which includes 48x48 grayscale face images with labels indicating their use (Training, PrivateTest, PublicTest).
    - Each image is represented as an array of pixel values, and the data is split into training, validation, and testing sets.
    - The images are resized to 128x128 and "salt and pepper" noise is added for the training process.

2. **Image Preprocessing**:
    - **Image Reconstruction**: Convert the grayscale images to RGB format and normalize the pixel values to the range [0, 1].
    - **Adding Noise**: Add "salt and pepper" noise with specific ratios to the images.

3. **Model Architecture**:

### 3.1 Generator (U-Net Architecture)
The generator uses a U-Net architecture with the following components:
   - **Encoder**: 
     - Consists of Conv2D layers, LeakyReLU activations, and BatchNorm2D to compress input images into lower-dimensional representations.
   - **Decoder**: 
     - Uses ConvTranspose2D layers to reconstruct the original image dimensions.
     - Skip connections between encoder and decoder preserve spatial information.
   - **Output Layer**:
     - A convolutional layer produces the final 3-channel image with normalized pixel values in the range [0, 1].

### 3.2 Discriminator (Convolutional Network)
The discriminator is a convolutional network that classifies images as "real" or "fake." It includes:
   - Multiple Conv2D layers with stride=2 for downsampling.
   - LeakyReLU activations for introducing non-linearity.
   - BatchNorm2D for stabilizing training.

4. **Training Process**:
   - **Loss Function**: The model uses MSELoss for training the generator.
   - **Optimizer**: Adam Optimizer with a learning rate of 0.001 for optimizing network parameters.
   - **Training Steps**:
     1. The noisy images are fed to the model, and the reconstructed output is generated.
     2. The reconstruction error between the clean and the reconstructed images is calculated, and the model is updated accordingly.
   - The best-performing model is saved during training.

5. **Evaluation Metrics**:
   - **PSNR (Peak Signal-to-Noise Ratio)**: Measures the similarity between clean and reconstructed images, achieving a value of 37.19.
   - **SSIM (Structural Similarity Index)**: Evaluates the structural similarity between images, achieving a value of 0.9774.
   - **MSE (Mean Squared Error)**: Measures pixel-wise error between clean and reconstructed images, achieving a value of 0.000193.

## Requirements
- Python 3.x
- TensorFlow or PyTorch (depending on the framework used)
- NumPy
- OpenCV
- Matplotlib

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository_url>

