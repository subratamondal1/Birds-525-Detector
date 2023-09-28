# [🕊️Birds 525 Species Detector🕊️](https://huggingface.co/spaces/iamsubrata/birds-525-species-detector)
---
## 1. Load Data
[**BIRDS 525 SPECIES IMAGE CLASSIFICATION DATASET**](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)
- **Dataset Size:** 525 bird species.
- **Images:** 84,635 training images, 2,625 test images (5 per species), and 2,625 validation images (5 per species).
- **Image Dimensions:** 224x224x3 JPG format.
- **Data Structure:** Convenient subdirectories for each species.
- **Additional File:** `birds.csv` contains file paths, class labels, and more.

**Dataset Quality:**
- Cleaned to remove duplicate and low-quality images.
- Each image features a single bird taking up at least 50% of pixels.
- Original images, no augmentation.

**File Naming:**
- Sequential numbering, e.g., "001.jpg," "002.jpg" for training.
- Test and validation images: "1.jpg" to "5.jpg."

**Note:** Training set not balanced but has ≥130 images per species.

## 2. Data Preparation
**Augmentation Transforms:**
- Imported augmentation transforms to enhance dataset diversity.

**Item Transforms:**
- Images resized to a consistent width of 460 pixels.

**Batch Transforms:**
- Images resized to 224x224 pixels with a minimum scaling of 75%.
- Additional augmentations applied to batches for robustness.

These transformations ensure data consistency and introduce variability, preparing the dataset for model training.

## 3. Create DataLoader
**DataBlock Configuration:**
- Utilized Fastai's DataBlock to define data processing.
- Blocks specified as (ImageBlock, CategoryBlock) for image and category data.
- `get_items` set to retrieve image files.
- Splitter configured using `GrandparentSplitter` to separate train and test sets based on directory names.
- `get_y` method defined to extract parent label as category.
- Item and batch transforms (`item_tfms` and `batch_tfms`) applied as configured.

**DataLoaders Creation:**
- Fastai DataLoaders created using the DataBlock and dataset path.
- Batch size set to 64 for efficient data loading.

These steps establish the DataLoaders required for training and validation, making the data ready for model ingestion.

## 4. Create Learner (Model) & Find Learning Rate
**Pre-trained Model Configuration:**
- Imported pre-trained ResNet models (resnet34 and resnet50) from torchvision.
- Created a vision learner using the Fastai library with the ResNet-50 architecture.
- Enabled the use of pretrained weights and specified evaluation metrics, including accuracy and error rate.
- Enabled mixed-precision training using `to_fp16()` for enhanced training efficiency.

**Finding Learning Rate:**
- Utilized the `lr_find()` method to determine an optimal learning rate.
- Discovered a suitable learning rate range (`lr`) for model training, set as `slice(1e-4, 5e-3)`.

This section highlights the model configuration, including the choice of architecture, pretrained weights, and learning rate discovery, ensuring an effective setup for model training and evaluation.

## 5. Train & Save Model
**Model Performance with Resnet50 (Freezed Layers)**

| Epoch | Train Loss | Valid Loss | Accuracy | Error Rate | Time   |
|-------|------------|------------|----------|------------|--------|
| 0     | 1.2802     | 0.4474     | 0.8705   | 0.1295     | 13:50  |
| 1     | 0.7865     | 0.1838     | 0.9482   | 0.0518     | 11:06  |
| 2     | 0.4920     | 0.1074     | 0.9695   | 0.0305     | 11:17  |
| 3     | 0.3435     | 0.0671     | 0.9844   | 0.0156     | 10:44  |
| 4     | 0.2979     | 0.0590     | 0.9859   | 0.0141     | 11:12  |

**Model Training:**
- The model was trained for 5 epochs using a one-cycle learning rate policy with the specified learning rate range.
- The training results demonstrate the model's impressive performance, achieving high accuracy and low error rates.

**Model Preservation:**
- Saved the trained model with the name 'model1_freezed' for future use.
- This step ensures that the model's architecture and learned weights are preserved and can be easily loaded and deployed for various tasks.

This section provides an overview of the model's training performance, including training and validation losses, accuracy, and error rates. It also emphasizes the importance of preserving the trained model for future use.



