# Lane Detection for Jetracer 🚗 by Jie Li, University of Nottingham, Department of Electrical and Electronic Engineering, AGPL-3.0 license
"""
Run advanced lane detection on Jetracer using the onboard camera of Jetson Nano. This script is specifically tailored for lane following on Jetracer's provided tracks.

Usage:
    $ python lanedet_jetson_nano.py                                   # Uses Jetson Nano's camera by default
                                                   --source img.jpg  # Processes a specific image
                                                   --source vid.mp4  # Processes a specific video
                                                   --source path/    # Processes all images in a directory

Preparation:
    Before running the script, ensure that comments related to Jetracer and Jetson Nano specific code are active if you are running on Jetracer platform. The script utilizes OpenCV and specific Nvidia libraries for optimal performance on Jetson Nano.

Features:
    - Utilizes GStreamer pipeline optimized for Jetson Nano to fetch video feed.
    - Implements dynamic region of interest and gradient thresholding for precise lane detection.
    - Uses Hough Transform for line detection and polynomial regression for lane curvature estimation.
    - Integrates PID control logic to adjust Jetracer's steering based on lane position.

Contact:
    - Jie Li
    - Email: jerrrjieli@outlook.com (Primary)
             872071077@qq.com (Secondary)
    - Former email (no longer in use post-graduation): ssyjl7@nottingham.ac.uk

Example commands:
    $ python lanedet_jetson_nano.py                                 # Automatically starts using Jetson Nano's camera
    $ python lanedet_realworld.py --source ./data/image/5.png  # Processes a specified image
    $ python lanedet_realworld.py --source ./data/videos/solidWhiteRight.mp4.mp4 # Processes a specified video
"""

import cv2
import numpy as np
import argparse
import os
import time

import sys
from pathlib import Path

# from jetracer.nvidia_racecar import NvidiaRacecar

# car = NvidiaRacecar()
error = 0


def gstreamer_pipeline(capture_width=1640, capture_height=1232, display_width=960, display_height=540, framerate=10,
                       flip_method=0):
    """
    返回适用于Jetson Nano的GStreamer管道字符串。
    """
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


class PIDControllers:

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.errorOld = 0.0
        self.errorSum = 0.0

    def PID(self, error):
        err = error

        self.errorSum += err

        proportional = err * self.kp

        integral = self.errorSum * self.ki

        differential = (err - self.errorOld) * self.kd

        output = proportional + integral + differential

        self.errorOld = err

        return output


# def calculate_dynamic_threshold(gray):
#     # 计算灰度直方图
#     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#     # 基于直方图的%分位数来确定阈值
#     cum_hist = np.cumsum(hist) / np.sum(hist)
#     threshold_index = np.where(cum_hist > 0.2)[0][0]
#     return threshold_index

def apply_gradient_threshold(image, sigma=1):
    # Calculate gradients in x and y direction using Sobel operator
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of gradients
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Compute the direction of gradients
    direction = np.arctan2(grad_y, grad_x)

    # Calculate threshold based on the standard deviation of gradient magnitudes
    std_dev = np.std(magnitude)
    threshold = np.mean(magnitude) + sigma * std_dev

    # Apply mask where the magnitude of the gradient exceeds the threshold
    mask = (magnitude > threshold).astype(np.uint8) * 255

    return mask, threshold, magnitude, direction

def apply_mask(edges, height, width):
    # Define polygonal mask for the edges
    poly_bottom_y = int(4 * height / 5)  # Bottom y-coordinate of the polygon
    poly_top_y = int(2 * height / 5)  # Top y-coordinate of the polygon
    center_x = width // 2  # Center x-coordinate of the image

    # Define polygons for left and right lane masking
    left_polygon = np.array([
        [(0, poly_bottom_y),
         (center_x - int(width / 3), poly_bottom_y),  # Shift left from center
         (center_x, poly_top_y),
         (int(width / 6), poly_top_y)]
    ], np.int32)

    right_polygon = np.array([
        [(center_x + int(width / 3), poly_bottom_y),  # Shift right from center
         (width, poly_bottom_y),
         (int(5 * width / 6), poly_top_y),
         (center_x, poly_top_y)]
    ], np.int32)

    # Create a mask and apply the polygons
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [left_polygon], 255)
    cv2.fillPoly(mask, [right_polygon], 255)

    # Apply the mask to the edges
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges, poly_bottom_y, poly_top_y, mask

def detect_lanes_hough_transform(masked_edges, frame, width):
    # Segment the image into left and right parts and detect straight lines using Hough transform.

    # Split the image into left and right parts
    left_edges = masked_edges[:, :width // 2]
    right_edges = masked_edges[:, width // 2:]

    # Lane detection based on Hough transform
    lines_left = cv2.HoughLinesP(left_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=50, maxLineGap=100)
    lines_right = cv2.HoughLinesP(right_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=50, maxLineGap=100)

    # Create an empty image for drawing lines
    line_image = np.zeros_like(frame)

    return line_image, lines_left, lines_right

def filter_lines_both_sides(lines_left, lines_right, line_image, width, slope_threshold=(0.5, 2), sample_step=5):
    # Filter and sample the points of the left and right line segments that meet the slope requirement.
    def filter_lines(lines, width_offset=0):
        points = []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if (x2 - x1) != 0:  # 防止除以零
                        slope = (y2 - y1) / (x2 - x1)
                        if slope_threshold[0] < abs(slope) < slope_threshold[1]:
                            num_points = int(np.hypot(x2 - x1, y2 - y1) // sample_step)
                            for i in np.linspace(0, 1, num=num_points):
                                x = int(x1 + (x2 - x1) * i) + width_offset
                                y = int(y1 + (y2 - y1) * i)
                                points.append((x, y))
                                cv2.circle(line_image, (x, y), 2, (255, 255, 0), -1)
        return points

    left_points = filter_lines(lines_left)
    right_points = filter_lines(lines_right, width // 2)

    return left_points, right_points

def fit_and_draw_polyline(left_points, right_points, line_image, height, poly_top_y, poly_bottom_y):
    left_fit_fn = right_fit_fn = left_top_point = right_top_point = None
    left_points = np.array(left_points)
    right_points = np.array(right_points)

    if len(left_points) > 1:
        left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
        left_fit_fn = np.poly1d(left_fit)
        left_top_point = (int(left_fit_fn(poly_top_y)), poly_top_y)

    if len(right_points) > 1:
        right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
        right_fit_fn = np.poly1d(right_fit)
        right_top_point = (int(right_fit_fn(poly_top_y)), poly_top_y)

    # Plotting polynomial curves
    for fit_fn, color in [(left_fit_fn, (0, 255, 0)), (right_fit_fn, (0, 255, 0))]:
        if fit_fn is not None:
            xs = fit_fn(np.linspace(poly_top_y, poly_bottom_y, 500))
            ys = np.linspace(poly_top_y, poly_bottom_y, 500)
            points = np.int32(list(zip(xs, ys)))
            cv2.polylines(line_image, [points], isClosed=False, color=color, thickness=5)

    # Connecting the tops of two curves
    if left_top_point and right_top_point:
        cv2.line(line_image, left_top_point, right_top_point, (0, 255, 0), 5)

    return left_fit_fn, right_fit_fn

def calculate_and_visualize_lane_center(frame, left_poly, right_poly, poly_top_y, poly_bottom_y, width, step=10,
                                        color=(255, 0, 0), thickness=2):

    # Calculate the lane median, plot it onto the image, and calculate the average deviation of the median from the centre of the image.

    if left_poly is None or right_poly is None:
        return frame, 0

    ys = np.arange(poly_top_y, poly_bottom_y, step)
    left_xs = left_poly(ys)
    right_xs = right_poly(ys)

    center_xs = (left_xs + right_xs) / 2
    lane_center_line = np.column_stack((center_xs, ys)).astype(np.int32)

    # Visualisation of the lane centre line
    for x, y in lane_center_line:
        cv2.circle(frame, (int(x), int(y)), radius=2, color=color, thickness=thickness)

    # Calculate the average of the distance from the centre of the lane to the centre of the image
    image_center_x = width // 2
    deviations = center_xs - image_center_x
    average_deviation = int(np.mean(deviations))

    # Text Setting
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    font_thickness = 2

    # Print the coordinates and deviation at the bottom
    cv2.line(frame, (image_center_x, poly_bottom_y), (image_center_x, poly_top_y), (0, 0, 255), 5)

    cv2.putText(frame, f'Error: {average_deviation}', (image_center_x, int((poly_top_y + poly_bottom_y) / 2)), font,
                font_scale * 2, font_color,
                font_thickness, cv2.LINE_AA)

    return frame, error


def detect_lane_lines(frame, previous_lines=None, show_steps=True):
    global error

    if show_steps:
        cv2.imshow('Original Image', frame)
    # RGB to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if show_steps:
        cv2.imshow('Gray Scale Image', gray)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if show_steps:
        cv2.imshow('Gaussian Blur', blur)

    # Dynamic calculation of Canny thresholds using DLD
    DLD_mask, threshold, magnitude, direction = apply_gradient_threshold(blur)
    if show_steps:
        cv2.imshow('Dynamic threshold Mask', DLD_mask)

    # Canny edge detection using DLD thresholds
    edges = cv2.Canny(DLD_mask, threshold / 2, threshold)
    # edges = cv2.Canny(blur, 50, 100)

    if show_steps:
        cv2.imshow('Canny Edges', edges)

    # Create mask
    height, width = frame.shape[:2]

    masked_edges, poly_bottom_y, poly_top_y, mask = apply_mask(edges, height, width)
    if show_steps:
        cv2.imshow('Masked Edges', masked_edges)
        cv2.imshow('Mask', mask)

    line_image, lines_left, lines_right = detect_lanes_hough_transform(masked_edges, frame, frame.shape[1])

    # Use filter_lines function
    left_points, right_points = filter_lines_both_sides(lines_left, lines_right, line_image, frame.shape[1])

    if show_steps:
        cv2.imshow('Hough Lines', line_image)

    # Perform a polynomial fit and plot the curve while obtaining the top point
    left_poly, right_poly = fit_and_draw_polyline(left_points, right_points, line_image, height, poly_top_y,
                                                  poly_bottom_y)

    if show_steps:
        cv2.imshow('Line Image', line_image)

    # Calculate the lane centre line, plot it on the image and obtain the mean deviation
    line_image, error = calculate_and_visualize_lane_center(line_image, left_poly, right_poly, poly_top_y,
                                                            poly_bottom_y, width)

    if show_steps:
        cv2.imshow('Error Calculation', line_image)

    # Image combination
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return result, error


def process_image(image_path, show_steps=True):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open/read file: {image_path}")
        return
    lane_image, _ = detect_lane_lines(image, show_steps=show_steps)
    cv2.imshow('Lane Lines Detection in Image', lane_image)
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def process_video(video_path, show_steps=True):
    if video_path == "gstreamer":
        # Using the GStreamer pipeline
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    else:
        # Use traditional file paths or camera indexes
        cap = cv2.VideoCapture(video_path)

    pid_controller_1 = PIDControllers(kp=0.002, ki=0.00001, kd=0.0)

    previous_lines = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lane_image, error = detect_lane_lines(frame, previous_lines, show_steps=show_steps)
        # previous_lines = lines  # Update previous lines for dynamic ROI adjustment
        cv2.imshow('Lane Lines Detection in Video', lane_image)

        pid_output_1 = pid_controller_1.PID(error)

        car_steering = max(-1, min(1, pid_output_1))
        # car.steering = car_steering

        # car.throttle = -0.4
        # print("car,throttle: ", car.throttle)

        if pid_output_1 > 0:
            print("Turn Right, car.steering=", car_steering)

        if pid_output_1 < 0:
            print("Turn Left, car.steering=", car_steering)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main(source):
    # car.steering_offset = 0
    # car.steering_gain = 1
    # car.throttle_gain = 1

    try:
        # Detect whether source is a picture, video or camera
        if source.isdigit():  # Camera
            print("Processing video from camera...")
            process_video("gstreamer", show_steps=True)
        elif os.path.isfile(source) and source.lower().endswith(('.png', '.jpg', '.jpeg')):  # Image file
            print("Processing an image...")
            process_image(source, show_steps=True)
        elif os.path.isfile(source):  # Video file
            print("Processing video file...")
            process_video(source, show_steps=False)
        else:
            print("The source argument does not match any supported type (image, video file, camera).")
    except KeyboardInterrupt:
        print("Interrupted! Stopping the car...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Lines Detection with Dynamic ROI and Preprocessing")
    parser.add_argument("--source", type=str, default='0',
                        help="Image or video file path, or camera index (e.g., 0 for the default camera)",
                        required=False)
    args = parser.parse_args()

    main(args.source)
