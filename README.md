# easy_CycleGAN

This repository contains two implementations of CycleGAN:

· CycleGAN.py: A standard CycleGAN implementation.

· cycleGAN.py: A CycleGAN implementation with gradient penalty.

# Project Overview

CycleGAN is a type of Generative Adversarial Network (GAN) that can perform image-to-image translation without requiring paired examples. This project includes a standard implementation of CycleGAN and an enhanced version with gradient penalty to improve training stability.

# Training

To train the CycleGAN model, use the following command: python CycleGAN.py

To train the CycleGAN model with gradient penalty, use: python cycleGAN.py

# Code Explanation

# Residual Block

The ResBlk class defines a residual block used in the generator network. It includes reflection padding, convolutional layers, instance normalization, and ReLU activation.

# Generator

The Generator class defines the generator network used in CycleGAN. It includes convolutional layers, instance normalization, residual blocks, and transposed convolutional layers for up-sampling.

# Discriminator

The Discriminator class defines the discriminator network used in CycleGAN. It includes convolutional layers, instance normalization, and LeakyReLU activation.

# Utility Function

The show_last_image function is used to display the last generated image during training.

The gradient_penalty function is used to calculate the gradient penalty.

# Training Loop

The training loop for the CycleGAN model, including the calculation of generator and discriminator losses, and the optimization steps.

# Result

Significant results can be achieved at epoch 20.

epoch 20

![Figure_1](https://github.com/user-attachments/assets/a4a31f8f-1fbe-45e8-9698-c7f51aea185b)




