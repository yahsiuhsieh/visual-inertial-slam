# SLAM using Extended Kalman Filter

This project aims to simultaneously localize a robot and map an unknown outdoor environment using IMU data and a 2D stereo camera features. An EKF based approach is taken to achieve the objective.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Please review [requirements.txt](https://github.com/arthur960304/visual-inertial-slam/blob/master/requirements.txt).

## Code organization

    .
    ├── docs                   # Folder contains robot and data specs
    ├── report                 # Folder contains my report analysis
    ├── results                # Folder contains final results images
    ├── src                    # Python scripts
    │   ├── main.py            # Main Visual-Intertial SLAM  file
    │   ├── slam.py            # Helper for Visual-Intertial SLAM
    │   ├── slam_utils.py	   # Utility sets of the of the SLAM
    │   └── visualize_utils.py # Utility sets to visualize the result
    └── README.md

## Running the tests

### Steps

1. Modify line 11 in `main.py` if you want to try different dataset.
2. Run the command `python main.py` and the resulting images will display.
3. You can change visualization funciton at line 24 to visualize different result, see more details in `visualize_utils.py`.

## Implementations

* See the [report](https://github.com/arthur960304/visual-inertial-slam/blob/master/report/report.pdf) for detailed implementations.

## Results

<p align="center">
  <img width="500" height="400" src="https://github.com/arthur960304/visual-inertial-slam/blob/master/results/compare22.png">
</p>


## Authors

* **Arthur Hsieh** - *Initial work* - [arthur960304](https://github.com/arthur960304)
