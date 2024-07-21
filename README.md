# Visual-Inertial-Odometry
This repository contains code for pose estimation using Camera and IMU sensor using both mathematical and deep learning approaches. For more details refer reports.

##1. VIO using Multi-State Constraint Kalman Filter filter
MSCKF (Multi-State Constraint Kalman Filter) is an EKF based **tightly-coupled** visual-inertial odometry algorithm. [S-MSCKF](https://arxiv.org/abs/1712.00036) is MSCKF's stereo version. This project is a Python reimplemention of S-MSCKF, the code is directly translated from official C++ implementation [KumarRobotics/msckf_vio](https://github.com/KumarRobotics/msckf_vio).  


For algorithm details, please refer to:
* Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight, Ke Sun et al. (2017)
* A Multi-State Constraint Kalman Filterfor Vision-aided Inertial Navigation, Anastasios I. Mourikis et al. (2006)  

## Requirements
* Python 3.6+
* numpy
* scipy
* cv2
* [pangolin](https://github.com/uoip/pangolin) (optional, for trajectory/poses visualization)

## Dataset
* [EuRoC MAV](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets): visual-inertial datasets collected on-board a MAV. The datasets contain stereo images, synchronized IMU measurements, and ground-truth.  
This project implements data loader and data publisher for EuRoC MAV dataset.

## Run  
`python vio.py --view --path path/to/your/EuRoC_MAV_dataset/MH_01_easy`  
or    
`python vio.py --path path/to/your/EuRoC_MAV_dataset/MH_01_easy` (no visualization)  

## Results
Trajectory Top View  
![](Code/VIO_Kalman/imgs/Trajectory_top_view.png)

Trajectory Side View
![](Code/VIO_Kalman/imgs/Trajectory_side_view.png)

## License and References
Follow [license of msckf_vio](https://github.com/KumarRobotics/msckf_vio/blob/master/LICENSE.txt). Code is adapted from [this implementation](https://github.com/uoip/stereo_msckf).

##2. Deep VIO, VO and IO
This phase contains Visual-Inertial Odometry, Visual Odometry and Inertial Odometry using Deep Neural Networks. Detailed network architecture and results are provided in a report in the Deep_VIO directory. 
Data Generation:
Synthetic images are generated by rendering a quadrotor in Blender simulation. Further, 8 different trajectories are generated to capture data. Some sample trajectories are shown below:

## Spiral Trajectory
![](Code/Deep_VIO/images/example_path.jpeg)

## Figure Eight Trajectory
![](Code/Deep_VIO/images/Fig_8.jpeg)

## Results
#Deep VIO Position Estimation 
![](Code/Deep_VIO/images/VIO_pos.jpeg)

#Deep VO Position Estimation
![](Code/Deep_VIO/images/VO_pos.jpeg)

#Deep IO Position Estimation
<img src="Code/Deep_VIO/images/IO_pos.jpeg" alt="Project Screenshot" width="500" height="300">
![](Code/Deep_VIO/images/IO_pos.jpeg)

