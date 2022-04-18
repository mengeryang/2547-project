# Reflection Removal Project for CSC2547

## Intoduction

This is a project of csc-2547. Based on the pytorch implementation of "[Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements](https://arxiv.org/abs/1904.00637)".


## Requirements

* Python >=3.5, PyTorch >= 0.4.1
* Requirements: opencv-python, tensorboardX, visdom


you can use `environment.yaml` to create your environment

## Datasets

### Training dataset
* 7,643 cropped images with size 224 × 224 from
  [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) (image ids are provided in VOC2012_224_train_png.txt, you should crop the center region with size 224 x 224 to reproduce our result). 

* 90 real-world training images from [Berkeley real dataset](https://github.com/ceciliavision/perceptual-reflection-removal) 

### Testing dataset
* 100 synthetic testing images from [CEILNet dataset](https://github.com/fqnchina/CEILNet) (testdata_reflection_synthetic_table2) 
* 20 real testing images from [Berkeley real dataset](https://github.com/ceciliavision/perceptual-reflection-removal).


      
## Usage
### Training

* Train the model by ```python train_errnet.py --name errnet --hyper --pixel_loss [mse+grad|ms_ssim_l1+grad|ms_ssim_l1|highpass] ``` , by changing the value of `pixel_loss`, you can train the model with different kind of loss functions.
* Check ```options/errnet/train_options.py``` to see more training options. 

### Testing

Evaluate the model performance by ```python test_errnet.py --name errnet -r --icnn_path [model-path] --hyper --testcase [1|2]```, set `testcase` to 1 for synthetic data or 2 for read data


