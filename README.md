<h1 align="center"><a href="https://huggingface.co/spaces/iamsubrata/birds-525-species-detector" target="_blank" rel="noopener noreferrer">Bird Species Detector</a></h1>

## Overview

The Bird Species Detector is a web application designed to classify bird species using machine learning. It allows users to upload or capture images of birds and uses a pre-trained model to predict the species.

## Features

- **Image Upload**: Users can upload images of birds for classification.
- **Camera Capture**: Users can capture images of birds using their device's camera.
- **Species Classification**: The application classifies the bird species and provides a confidence level for the prediction.
- **Model Information**: Displays information about the model's performance, including accuracy and error rate.
- **Dataset**: Utilizes the "BIRDS 525 SPECIES IMAGE CLASSIFICATION DATASET" with 84,635 training images, 2,625 test images, and 2,625 validation images across 525 bird species.

## Technical Details

### Data Preparation

- **Augmentation**: Applies augmentation transforms to enhance dataset diversity.
- **Transforms**: Resizes images and applies batch transforms for model training.

### Model Training

- **Architecture**: Uses pre-trained ResNet models (resnet34 and resnet50) from torchvision.
- **Learning Rate**: Employs the `lr_find()` method to determine an optimal learning rate.
- **Training**: Conducts training using a one-cycle learning rate policy.

### Web Application

- **Framework**: Built with Streamlit.
- **Functions**: Includes `upload_photo`, `capture_photo`, and `model_info` to handle user interactions.
- **User Interface**: Provides a friendly UI with options to upload, capture, and view model information.

## Challenges and Solutions

### Challenge 1: Data Imbalance

The dataset was not balanced, which could lead to biased predictions favoring more common species.

**Solution**: Implemented data augmentation techniques to artificially expand the dataset, providing more examples for underrepresented classes.

### Challenge 2: Model Performance

Initial models struggled with high error rates and low accuracy.

**Solution**: Utilized transfer learning with pre-trained ResNet models and fine-tuned the learning rate to improve model performance significantly.

### Challenge 3: User Experience

Creating an intuitive user interface that allows non-technical users to interact with the machine learning model.

**Solution**: Chose Streamlit for the frontend to create a simple yet powerful interface with options to upload images, capture photos, and display model predictions and information.

### Challenge 4: Deployment

Deploying a machine learning model in a web application can be complex due to the need for high computational resources.

**Solution**: Optimized the model for inference and used Streamlit sharing for deployment, which abstracts away the complexities of hosting a machine learning model.

## Conclusion

The Bird Species Detector is a testament to the power of machine learning in the field of biodiversity and conservation. By leveraging advanced deep learning techniques and an intuitive web interface, this application makes species classification accessible to a wide audience.

---
