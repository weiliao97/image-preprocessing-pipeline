# Experimental image processing 

Multichannel fluorecent images were acquired by a Nikon Eclipse Ti microscope using the following imaging specifications:
- 20x, PH1 
- Phase: 50 ms. 3.5V, FITC : 200 ms 
- Capture 10x10 field using ND acquisition 

The pipeline here read the ND file and perform 
- Illumination correction
- Single cell detection (using opencv)
- Watershed to extract foreground ana background 
- Perform quantification on the fluorescent image 

The output phase contrast and fluorescent images can be used in the downstream deep learning models. 


![Phase and FITC](https://drive.google.com/file/d/1jI7Q1g9ZqzGlBrQeBiNkrg-fUpavfDbO/view?usp=sharing)
