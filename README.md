# Siamise_network
## Introduction
> This repo is implement an siamese network for one-shot learning with resnet50 in backbone and triplet loss
## Dependencies
> python > 3.6
> pip newest version
## Usage
> cd root && pip3 install -r requirements.txt
> To run train.py you need config the path of training dataset in line 26 & 28.
> Format folder training is
> Data_shopee
>  |
>  |----->train_images_triplest.csv
>  |----->train_images
> Format csv file training is
> anchor	postive		negative
> img_name	img_name	img_name
