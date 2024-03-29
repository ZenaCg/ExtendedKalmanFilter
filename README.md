## Introduction

Simultaneous Localization and Mapping (SLAM) is an important topic within the mobile robotics community. The solutions to the SLAM problem has brought possibilities for a mobile robot to be placed at an unknown location in an unknown environment and simultaneously building a map and locating itself. As it means to make robots truly autonomous, the published SLAM algorithm are widely employed in self-driving cars, autonomous underwater vehicles and unmanned aerial vehicles. In this project, within motion measurements provided by an inertial measurement unit and visual features with precomputed correspondences by stereo cameras, we design an extended Kalman filter to recursively provide the best estimate of localization and mapping.

## Data

- Synchronized measurements from an IMU and a stereo camera are used.

## Code

- Python script: EKF.py : main function of the extended Kalman filter.
- Python script: proj_geometry.py : helper functions to perform calculations for SO(3) and SE(3).
- Python script: utils.py : helper functions for data loading and result visualization.
