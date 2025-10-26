
## HW3 - CSCI4931

**Student:** Shivam Pathak  
**Assignment:** Hat Detection Competition  

## Plots from code 
![alt text](image.png)
![alt text](image-1.png)
## Problem
Using Binary classification to detect if people in images wear hats. Classes: "Hat" vs "No Hat".

## Dataset & Preprocessing
- **Source:** CelebA with "Wearing_Hat" attribute
- **Size:** 6,000 balanced images (3k each class)
- **Split:** 80% train, 20% validation
- **Preprocessing:** 160×160 resize, normalize to [0,1]
- **Augmentation:** Horizontal flip, brightness/contrast adjustment

## Model Architecture
- **Base:** MobileNetV2 (ImageNet pre-trained)
- **Head:** Global Average Pooling → Dropout(0.2) → Dense(1, sigmoid)
- **Training:** Two-phase approach
  1. Frozen base, 6 epochs
  2. Fine-tune last 20 layers, 3 epochs, lr=1e-5

## Solution Evolution 
Initially I started with a simple CNN to confirm that the dataset pipeline was working, but the accuracy only went up to around 70%. Then I shifted to transfer learning using the MobileNetV2 with ImageNet weights, this improved the training stability and validation accuracy. 

After verifying that the base layers were frozen, I went ahead and added a second training phase to fine-tune the top 20 layers with a lower learning rate (1e-5).

This two phase approach helped me increase the accuracy by 10% and helped the model generalize better to the Kaggle test set. 

## Results
- Successfully trained the model
- Generated competition submission
- Created training visualizations (loss.png, acc.png)
- Model converged with early stopping

The final model was around 95% validation accuracy and AUC of 0.99 on the held-out validation split. In the Kaggle competition my submission recorded around 83% accuracy, which was ranked 1 at the time of my submission. This confirmed for me that the model can generalize well from CelebA to the competition dataset. 

## Implementation
- **Framework:** TensorFlow/Keras
- **Environment:** Kaggle Notebooks
- **Reproducible:** Fixed seeds, documented dependencies

Overall this was a great homework that helped show me the importance of transfer learning for small custom datasets. Fine-tuning and data balancing had a much larger impact than I thought! 
