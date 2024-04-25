# Advanced Lane Detection Systems

## Description
This repository contains two advanced lane detection systems designed for different platforms:
1. **Lane Detection for Jetracer**: Tailored for Jetracer using the onboard camera of the Jetson Nano, specifically for lane following on Jetracer's provided tracks.
2. **Lane Detection Real World**: Designed to run on general-purpose computers and utilizes various sources like webcams, images, videos, and streaming URLs for lane detection.

## Author
- **Jie Li**
- **University of Nottingham, Department of Electrical and Electronic Engineering**

## Contact
- **Email**: jerrrjieli@outlook.com (Primary)
- **Secondary Email**: 872071077@qq.com
- **Former Email**: ssyjl7@nottingham.ac.uk (No longer in use post-graduation)

## License
Both projects are licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Installation

### Prerequisites
- For the Jetracer project: NVIDIA Jetson Nano
- For the Real World project: Any general-purpose computer with a compatible webcam or video input.
- Python 3.x
- OpenCV, tested with version 4.x

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repository/advanced-lane-detection.git
   cd advanced-lane-detection
   ```
2. **Install required Python packages:**
```bash
pip install opencv-python
```
3. ***For Jetracer on Jetson Nano***
Run the script using the Jetson Nano’s onboard camera by default:
```bash
python lanedet_jetson_nano.py
```
Run the script using test image:
```bash
python lanedet_jetson_nano.py --source data/images/lane7.png
```
4. ***For Real World Detection***
Run the script using the default image or specify a source:
```bash
python lanedet_realworld.py                                 # Default image
python lanedet_realworld.py --source ./Data/5.png  # Specific image
python lanedet_realworld.py --source ./Data/Videos/solidWhiteRight.mp4    # Specific video
python lanedet_realworld.py --source 0                      # Webcam
python lanedet_realworld.py --source ./Data/dataset  # Specific dataset
```

### Configuration
Ensure to activate specific code related to Jetracer and Jetson Nano in the script as per your setup for the Jetracer project.

### Features
1. Utilizes GStreamer pipeline optimized for Jetson Nano in the Jetracer project.
2. Dynamic region of interest and gradient thresholding for precise lane detection in both projects.
3. Hough Transform for line detection and polynomial regression for lane curvature estimation.
4. PID control logic to adjust Jetracer's steering based on lane position in the Jetracer project.
5. ***'apply_gradient_threshold()' function can take lots of calculation for jetson nano in real time detection, 'calculate_dynamic_threshold()' function can be an alternative method of calculating dynamic threshold***

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### Acknowledgments
University of Nottingham,
Department of Electrical and Electronic Engineering
