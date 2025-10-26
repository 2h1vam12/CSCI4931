# PA3 - Hat Detection Deep Learning Model

**Name:** Shivam Pathak 
**Class:** Deep Learning Fall 2025 (Professor Ashis Biswas) 
**Assignment:** HW3 - Hat or No Hat Competition  
**Date:** 10/26/25

## Overview
Deep learning solution for Kaggle competition "Hat or no hat, that is the question. Binary classification to detect if people in images are wearing hats.

## Dataset
- **Training:** CelebA dataset with "Wearing_Hat" attribute
- **Balanced subset:** 6,000 images (3,000 Hat, 3,000 No Hat)
- **Split:** 80% train, 20% validation

## Model
- **Architecture:** MobileNetV2 + transfer learning
- **Training:** Two-phase (frozen base → fine-tuning)
- **Input:** 160x160 RGB images
- **Output:** Binary classification (Hat/No Hat)

## How to Run
1. Go on Kaggle and add the following dataset as the input : https://www.kaggle.com/datasets/jessicali9530/celeba-dataset 
2. Install dependencies: `pip install -r requirements.txt`
3. Run all cells in `dlhw3codepathak.ipynb`
Note: GPU P100/T4 is recomended for training efficiency.
4. Generates `submission.csv` for Kaggle submisson 

## Files
- `dlhw3codepathak.ipynb` - Main code
- `requirements.txt` - Dependencies
- `README.md` - This file
- `report.pdf` - Detailed report

## External Resources
- CelebA dataset by Jessica Li: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
- MobileNetV2 (ImageNet pre-trained)
- TensorFlow/Keras
