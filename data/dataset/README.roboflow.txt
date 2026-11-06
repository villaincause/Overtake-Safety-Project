
Traffic Car Segmentation 2 - v2 2026-10-05 1:17pm
==============================

This dataset was exported via roboflow.com on October 14, 2026 at 3:58 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1052 images.
Traffic-cars-pSEL are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 384x384 (Stretch)
* Auto-contrast via contrast stretching

The following augmentation was applied to create 2 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -15 and +15 degrees
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -10 and +10 percent
* Random Gaussian blur of between 0 and 0.7 pixels
* Salt and pepper noise was applied to 0.1 percent of pixels


