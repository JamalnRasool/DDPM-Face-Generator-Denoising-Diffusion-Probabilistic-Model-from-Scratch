# DDPM Face Generator 

A Denoising Diffusion Probabilistic Model (DDPM) built from scratch using pure PyTorch, trained on the CelebA-HQ dataset to generate human faces.

## What is this?

This model starts from pure random noise and gradually denoises it over 300 timesteps to generate a brand new face image that never existed before.

## Model Details

- **Architecture:** U-Net with residual blocks and sinusoidal time embeddings
- **Channel progression:** 64 → 128 → 256
- **Noise schedule:** Cosine schedule with T=300 timesteps
- **Dataset:** CelebA-HQ (high quality face images)
- **Image size:** 128×128
- **Training:** 125+ epochs on dual NVIDIA T4 GPUs
- **Optimizer:** AdamW with cosine LR scheduler
- **Loss:** Mean Squared Error (MSE) on predicted noise

## How to use

1. Move the **Random Seed** slider to get different images
2. Click **Generate Image**
3. Watch the denoising steps on the left and the final generated face on the right

## Results

- PSNR Score: 5.38 dB
- SSIM Score: 0.0761

## Built with

- PyTorch
- Gradio
- Trained on Kaggle with dual T4 GPUs

## Authors

- Jamal Rasool
- Umar Zahoor

## Link
https://huggingface.co/spaces/rek49/genaiass4

