# Lane Detection for Jetracer

## Description
This project is designed to run advanced lane detection on Jetracer using the onboard camera of the Jetson Nano. It is specifically tailored for lane following on Jetracer's provided tracks and utilizes OpenCV along with Nvidia's specific libraries for optimal performance on Jetson hardware.

## Author
- **Jie Li**
- **University of Nottingham, Department of Electrical and Electronic Engineering**

## Contact
- **Email**: jerrrjieli@outlook.com (Primary)
- **Secondary Email**: 872071077@qq.com
- **Former Email**: ssyjl7@nottingham.ac.uk (No longer in use post-graduation)

## License
This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Installation

### Prerequisites
- NVIDIA Jetson Nano
- Python 3.x
- OpenCV, tested with version 4.x
- Access to Jetracer's hardware setup

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repository/lane-detection-jetracer.git
   cd lane-detection-jetracer
2. **Install required Python packages:**
pip install -r requirements.txt
3. **Usage**
Run the script using the Jetson Nano’s onboard camera by default:
   ```bash
   python lanedet_jetson_nano.py
   ```
To process a specific image:
   ```bash
   python lanedet_jetson_nano.py --source ./data/test_image.jpg
   ```
To process a video file:
   ```bash
   python lanedet_jetson_nano.py --source ./videos/test_video.mp4
   ```
4. **Configuration**
Make sure to activate specific code related to Jetracer and Jetson Nano in the script as per your setup.

5 **Features**
1. Utilizes GStreamer pipeline optimized for Jetson Nano.
2. Dynamic region of interest and gradient thresholding for precise lane detection.
3. Hough Transform for line detection and polynomial regression for lane curvature estimation.
4. PID control logic to adjust Jetracer's steering based on lane position.


6. **Contributing**
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

7. **Acknowledgments**
University of Nottingham
Department of Electrical and Electronic Engineering
