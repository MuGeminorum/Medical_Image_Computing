# MSSC
Medical Signal Segmentation and Classification

[![license](https://img.shields.io/github/license/george-chou/MSSC.svg)](https://github.com/george-chou/MSSC/blob/master/LICENSE)
[![Python application](https://github.com/george-chou/MSSC/workflows/Python%20application/badge.svg)](https://github.com/george-chou/MSSC/actions)
[![Github All Releases](https://img.shields.io/github/downloads-pre/george-chou/MSSC/v1.1/total)](https://github.com/george-chou/MSSC/releases)

## MRI Dataset

The MRI data, D, is stored as a 157-by-189-by-68 NumPy array. You can show image of each of the frame within the dataset using the imshow() function imported from matplotlib.pyplot. Please use this function to show the data in the axial view at slice 16, the data in the sagittal view at slice 64, and the data in the coronal view at slice 64. And set the aspect of the axis scaling (the ratio of y-unit to x-unit) to 0.5 when plotting the images.

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f1.png"/><br>
<b>Figure 1: Code for task 1</b>
</div>

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f2.png"/><br>
<b>Figure 2: Results of task 1</b>
</div>

## Edge Filters

Edge provides critical information about the shape of the region of interest (ROI) and serves as an important step in many segmentation algorithms. In this task, we work on the data in the axial view at slice 16 for edge detection. 
 
(1)	Please calculate the image gradients along x and y directions with the function cv2.Sobel(), and the gradient magnitude using Numpy.sqrt() function, and plot image gradients ("Gx", "Gy") as well as gradient magnitude ("Gmat") using imshow() function. 
 
(2)	Please define the Prewitt kernels in both x and y directions, and use the cv2.filter2D() function to complete prewitt edge filtering. 
 
(3)	Please conducting a Canny edge detection with the function of feature.canny() and plot the results using imshow() function. Please briefly explain Canny edge detection. What are the values and meaning of the double thresholds used in your canny edge detection? 
Please change the lower and upper thresholds to (2, 5) and (3, 15), respectively; How does the edge detection change? 
 
(4)	What is spatial frequency? Are edge filters low-pass filtering operation or high-pass filtering operation?

### Answer of (1)

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f3.png"/><br>
<b>Figure 3: Code for task 2(1)</b>
</div>

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f4.png"/><br>
<b>Figure 4: Results of task 2(1)</b>
</div>

### Answer of (2)

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f5.png"/><br>
<b>Figure 5: Code for task 2(2)</b>
</div>

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f6.png"/><br>
<b>Figure 6: Results of task 2(2)</b>
</div>

### Answer of (3)

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f7.png"/><br>
<b>Figure 7: Code for task 2(3)</b>
</div>

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f8.png"/><br>
<b>Figure 8: Results of task 2(3)</b>
</div>

### Answer of (4)

The spatial frequency is an independent variable to describe the characteristics of an image. It can decompose the spatial change of an image pixel value into a linear superposition of simple vibration functions with different amplitudes, spatial frequencies and phases. The composition and distribution of this spatial frequency component is called the spatial frequency spectrum. 
 
Edge filters belong to high-pass filtering operation.

## Kmeans clustering

(1)	Please describe the process of k-means clustering. What is the difference between supervised and unsupervised methods? What are their advantages and disadvantages? 
 
(2)	Import KMeans from the sklearn.cluster package. 
 
(3)	Please group the pixels in slice 16 using the function of KMeans(). Show the results with 4, 8, 20 clusters, respectively. 
 
To show the results, please replace the intensity value at each pixel with the intensity value of its corresponding cluster centres and display the resulting image use the function imshow(). This will give each cluster a unique colour. Please compare the results. 
 
(4)	If you repeat the algorithm for several times, will the results change? 
 
(5)	If 4 clusters are to be generated, please plot the relationship between withincluster sums of point-to-centroid distances (kmeans.inertia) and number of iterations (kmeans.n_iter). The x-axis corresponds to the number of iterations and the y-axis corresponds to the within-cluster sums. 

### Answer of (1)

Basic steps of k-means algorithm: 
(1)	Select k objects from the data as the initial cluster centres; 
(2)	Calculate the distance of each cluster object to the cluster centre to divide; 
(3)	Calculate each cluster centre again; 
(4)	Calculate the standard measurement function. If the method reaches the maximum number of iterations, stop, otherwise, continue the operation. 

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f9.png"/><br>
<b>Figure 9: Process of K-means clustering</b>
</div>

<div align=center>
<b>Table 1: Differences between supervised and unsupervised methods</b><br>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/t1.PNG"/>
</div>

<div align=center>
<b>Table 2: Advantages and disadvantages of supervised and unsupervised methods</b><br>
<img width="605" src="https://george-chou.github.io/covers/HEp-2/t2.PNG"/>
</div>

### Answer of (2)

'from sklearn.cluster import KMeans'

### Answer of (3)

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f10.png"/><br>
<b>Figure 10: Code for task 3(3)</b>
</div>

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f11.png"/><br>
<b>Figure 11: Results of task 3(3)</b>
</div>

### Answer of (4)

The results slightly change after the algorithm is repeated for several times:

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f12.png"/><br>
<b>Figure 12: Results of task 3(4)</b>
</div>

### Answer of (5)

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f13.png"/><br>
<b>Figure 13: Code for task 3(5)</b>
</div>

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f14.png"/><br>
<b>Figure 14: Results of task 3(5)</b>
</div>

## Support Vector Machine

Support-vector machines (SVMs) are supervised learning machine learning models widely used for classification and regression tasks. In medical research, SVMs can be used to predict the health status of a patient for a target disease. In this experiment, we are going to train an SVM model to predict Diabetes. 
 
Please download the pima.csv file from Canvas and save it to the same directory as this Jupyter notebook and run the following code to load the datasets. 

<div align=center>
<img width="605" src="https://george-chou.github.io/covers/MSSC/f15.png"/><br>
<b>Figure 15: Code for task 4</b>
</div>

'Output: 0.6771653543307087'