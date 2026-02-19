# CNN Satellite Image Classifier

This project uses a CNN to classify satellite images into 4 categories: Cloudy, Desert, Green Area, and Water. This project can be useful for monitoring the environment including desertification and farm outputs.

# Dataset Description
- 4 classes
  - Cloudy
  - Desert
  - Green Area
  - Water
- 5,631 images
- Image size: 250x250

Dataset link: https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification

# Setup

1. Install all required libraries
```
pip install -r requirements.txt
```

2. Run the jupyter notebook
```
cd /notebooks
jupyter notebook cnn_model.ipynb
```
#  Results

## Baseline CNN
Train Accuracy:      0.9487 (94.9%)

Validation Accuracy: 0.9174 (91.7%)

Gap (overfit check): 0.0313 (3.1%)

## Improved CNN with Augmentations
Train Accuracy:      0.7738 (77.4%)

Validation Accuracy: 0.7131 (71.3%)

Gap (overfit check): 0.0607 (6.1%)

# Sample Predictions
<img width="1332" height="393" alt="image" src="https://github.com/user-attachments/assets/f60cf670-4595-4a84-a579-2743bb5df459" />

# Team Member Contributions
Nicky Cheng: EDA, Report, and README
Sepehr Mansouri: CNN model and analysis
