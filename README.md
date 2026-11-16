# Carvana Image Segmentation

The task is binary segmentation - predict a mask that separates each vehicle from its background.

We will be using Python 3.11 for a majority of the project. The dependencies will be stored in `requirements.txt`


## Dataset split

Since the original Kaggle provided test dataset contains unlabelled datapoints (no masks), we had to take the original training dataset (which is labelled) of 5,088 images and split it into test and validation sets.

The training set contains 3,552 images, and the test and validation contain 768 images each.

The "naive" split where images are split randomly is in data/splits/naive, and the proper datasplit (where all 16 images of the same car are only in one of the sets and not the other) are in data/splits.