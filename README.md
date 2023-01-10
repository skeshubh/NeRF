# NeRF
An implementation of NeRF for novel view synthesis.  
This project was done in collaboration with [Dushyant Patil](https://github.com/dushyant6) and under the guidance of [Prof. Nitin J. Sanket](https://nitinjsanket.github.io/index.html)

## What is NeRF?
A neural radiance field (NeRF) is a fully-connected neural network that can generate novel views of complex 3D scenes, based on a partial set of 2D images. The method relies on capturing camera poses for images of a fixed object and finding a RGB map and opacity map to recreate a 3D representation. The steps to create NeRF are as follows:  
* Data generation using COLMAP  
* 3D ray generation and volume sampling  
* Radiance field estimation using DL model
* Final image and video generation  

#### Data generation using COLMAP:
The data generation was done using 100 images of a backhoe-loader toy taken from different angles on a white background without reflections. The COLMAP package was used to find the camera properties and poses. The camera pose data, global C-sys, rotation quaternions, translation vectors, and focal length were determined with the help of COLMAP (As shown in Fig.14). The Cartesian coordinates and rotation matrices were converted to spherical coordinates to represent the pose of camera (x, y, z, θ, ϕ).  

#### 3D ray generation and volume sampling:
The camera pose matrix and focal length found in previous step are used to generate rays passing through every pixel of the image to the global coordinate system origin. Using the volume rendering technique mentioned in the [original paper](https://arxiv.org/abs/2003.08934), a set of 16 points were placed along each 3D rays. These query points are generated for all pixel to estimate the depth of the object from the camera. The querypoints in 5D are used to generate an RGB map and a density map of the scene using the deep learning architecture as follows:  
![Camera pose estimation using COLMAP](Assets/Images/CameraPoseEstimationUsingCOLMAP.png)  
Having the network Fθ directly operate on xyzθϕ input coordinates results in renderings that perform poorly at representing high-frequency variation in color and geometry. Thus, the X,Y,Z out of the 5D points are converted to higher 60-dimensions and the θ and ϕ are converted to 24-dimensions before inputting them to the model. This mapping from R to R2L is done using positional encoding as follows:  
