# MIC
Medical Image Computing Module Development

[![license](https://img.shields.io/github/license/MuGeminorum/Medical_Image_Computing.svg)](https://github.com/MuGeminorum/Medical_Image_Computing/blob/master/LICENSE)
[![Python application](https://github.com/MuGeminorum/Medical_Image_Computing/workflows/Python%20application/badge.svg)](https://github.com/MuGeminorum/Medical_Image_Computing/actions)
[![Github All Releases](https://img.shields.io/github/downloads-pre/MuGeminorum/Medical_Image_Computing/v1.1/total)](https://github.com/MuGeminorum/Medical_Image_Computing/releases)
[![](https://img.shields.io/badge/wiki-mic-3572a5.svg)](https://github.com/MuGeminorum/Medical_Image_Computing/wiki/Chapter-I-%E2%80%90-Medical-image-computing)

## Medical Image Enhancement (MIE) ##

The source code of MIE is in _MedicalImageEnhancement.py_.

_Table 1_ compares the three filters, including smoothing, sharpening, edge detection in the aspects of the kernel, experiment demo, and time cost of running. The way of achieving a filter is to use the filter to perform convolution operations on 3D images.

<div align=center>
    <b>Table 1: Comparison of different filters</b><br>
    <img width="455" src="https://user-images.githubusercontent.com/20459298/233113143-f6f0b426-17f5-4b38-8d13-d7ae4af03a9a.PNG"/>
</div>

## Medical Image Segmentation (MIS) ##

The source code of MIS is in _MedicalImageSegmentation.py_.

_Figure 1_ shows the demonstration of 3D segmentation results achieved using three multiple viewing angles.

<div align=center>    
    <img width="100%" src="https://user-images.githubusercontent.com/20459298/233113218-15beccc4-0f79-4cea-a283-9e3956de3ee2.png"/><br>
    <b>Figure 1: 3D segmentation result of the tumor</b>
</div>

_Table 2_ shows the demonstration of the experiments on different global and local parameter combinations. 

<div align=center>
    <b>Table 2: Part experiments on different global and local parameters</b><br>
    <img width="455" src="https://user-images.githubusercontent.com/20459298/233113328-43869ae5-f62f-42bd-9808-a836d714089a.PNG"/>
</div>

Based on experiment results, the best global and local parameters are among experiments with ID from 2 to 3 considering whether the classification is true or false, positive or negative. The best way to find them is to define an indicator of the best result, then define a distance between the result indicator in current parameters and the best outcome, and finally search for the settings of minimum distance by deep learning. In that way, we do not need to search for the best parameters manually. 
