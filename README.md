# Experimental image processing 

Multichannel fluorecent images were acquired by a Nikon Eclipse Ti microscope using the following imaging specifications:
- 20x, PH1 
- Phase: 50 ms. 3.5V, FITC : 200 ms 
- Capture 10x10 field using ND acquisition 

The pipeline here read the ND file and perform 
- Illumination correction
- Single cell detection (using opencv)
- Watershed to extract foreground and background 
- Perform quantification on the fluorescent image 

The output phase contrast and fluorescent images can be used in the downstream deep learning models. 

## Original images 
Here is a sample of a pair of original phase contrast and FITC images. 

![Phase and FITC](/images/pair.png "Phase and FITC")

## Step 1. Illumination correction 

Illumination correction follows the paper Singh et al. [Pipeline for illumination correction of images for high-throughput microscopy](https://onlinelibrary.wiley.com/doi/10.1111/jmi.12178). Here is the original image, correction image in the FITC channel and after correction. 

![Correction](/images/correction.png "Correction")

## Step 2. Single cell detection 

Single cell detection was done bases on the intensity threshed on the FITC channel. 

## Step 3. Single cell segmentation 

Segmentation mask was created using watershed algorithm. Here are some examples. 
![Watershed](/images/watershed.png "Watershed")

## Step 4. ROS quantificaiton 
The mask will be used to calculate the average FITC intensity in the cell region (corrected by the intensity in the backgroud region), which is our ROS label. 
![ROS](/images/Figure 2021-03-31 080234.png "ROS")
