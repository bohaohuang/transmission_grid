# Power Grid Mapper

---
## Download the dataset
[Electric Transmission and Distribution Infrastructure Imagery Dataset](https://figshare.com/articles/dataset/Electric_Transmission_and_Distribution_Infrastructure_Imagery_Dataset/6931088)

## Environment Setup
`conda env create -f environment.yml`

## Preprocess the data
1. Make TFRecord for FasterRCNN
`python data/make_patches`
2. Make segmentation map for StackNet
`python make_angle_map.py`
   
## Train the model
1. FasterRCNN
Please refer to [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
2. StackNet
`python train_stack.py`
