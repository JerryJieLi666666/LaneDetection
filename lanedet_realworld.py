# Lane Detection 🚗 by Jie Li, University of Nottingham, Department of Electrical and Electronic Engineering, AGPL-3.0 license
"""
Run lane detection inference on images and videos using advanced computer vision techniques developed by Jie Li from the University of Nottingham's Department of Electrical and Electronic Engineering.

Usage - sources:
    $ python lanedet_realworld.py                                      # default image
                                                     --source img.jpg                         # image
                                                     --source vid.mp4                         # video
                                                     --source path/                           # directory
                                                     --source 'path/*.jpg'                    # glob
                                                     --source 0                               # webcam (camera index)
                                                     --source 'https://example.com/video.mp4' # HTTP stream

Usage - details:
    This script supports lane detection on various sources such as static images, video files,
    directories of images, image globs, webcams, and streaming URLs. It utilizes OpenCV for image
    processing operations, such as gradient thresholding for edge detection, applying masks for
    ROI specification, and utilizing Hough Transform for line detection. The script also features
    advanced polynomial fitting to estimate lane curvature and calculate vehicle position relative to lane center.

    Results from the processing can be visualized in real-time, and processed outputs are saved in a designated results folder.

Contact:
    - Jie Li
    - Email: jerrrjieli@outlook.com (Primary)
             872071077@qq.com (Secondary)
    - Former email (no longer in use post-graduation): ssyjl7@nottingham.ac.uk

Example commands:
    $ python lanedet_realworld.py                                 # Runs detection on the default image set in the script
    $ python lanedet_realworld.py --source ./data/image/5.png  # Processes a specified image
    $ python lanedet_realworld.py --source ./data/videos/solidWhiteRight.mp4.mp4 # Processes a specified video
    $ python lanedet_realworld.py --source ./data/dataset # Processes a dataset
    $ python lanedet_realworld.py --source 0                       # Processes video feed from the default webcam
"""


import cv2
import numpy as np
import argparse
import os


def apply_gradient_threshold(image, sigma=3):
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

def apply_mask(edges, height, width, center_offset):
    # Define polygonal mask for the edges
    poly_bottom_y = int(3 * height / 4)  # The y-coordinate of the bottom of the polygon
    # poly_top_y = int(2 * height / 3)  # The y-coordinate of the top of the polygon
    poly_top_y = int(2.5 * height / 5)  # The y-coordinate of the top of the polygon
    center_x = int(width // 2) - center_offset                 # Center x-coordinate of the image

    # Define polygons for left and right lane masking
    left_polygon = np.array([
        [(int(width / 3), poly_bottom_y),
         (center_x - int(1 * width / 20), poly_bottom_y),  # Shift left from center
         (center_x, poly_top_y),
         (center_x - int(width / 20), poly_top_y)]
    ], np.int32)

    right_polygon = np.array([
        [(center_x + int(1 * width / 20), poly_bottom_y),  # Shift right from center
         (int(2 * width / 3), poly_bottom_y),
         (center_x + int(width / 20), poly_top_y),
         (center_x, poly_top_y)]
    ], np.int32)

    # Create a mask and apply the polygons
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [left_polygon], 255)
    cv2.fillPoly(mask, [right_polygon], 255)

    # Apply the mask to the edges
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges, poly_bottom_y, poly_top_y, mask

# def apply_mask(edges, height, width):
#     # Define polygonal mask for the edges
#     poly_bottom_y = int(11 * height / 12)  # The y-coordinate of the bottom of the polygon
#     # poly_top_y = int(2 * height / 3)  # The y-coordinate of the top of the polygon
#     poly_top_y = int(3 * height / 5)  # The y-coordinate of the top of the polygon
#     center_x = width // 2                # Center x-coordinate of the image
#
#     # Define polygons for left and right lane masking
#     left_polygon = np.array([
#         [(int(width / 7), poly_bottom_y),
#          (center_x - int(width / 5), poly_bottom_y),  # Shift left from center
#          (center_x, poly_top_y),
#          (center_x - int(width / 12), poly_top_y)]
#     ], np.int32)
#
#     right_polygon = np.array([
#         [(center_x + int(width / 5), poly_bottom_y),  # Shift right from center
#          (int(6 * width / 7), poly_bottom_y),
#          (center_x + int(width / 12), poly_top_y),
#          (center_x, poly_top_y)]
#     ], np.int32)
#
#     # Create a mask and apply the polygons
#     mask = np.zeros_like(edges)
#     cv2.fillPoly(mask, [left_polygon], 255)
#     cv2.fillPoly(mask, [right_polygon], 255)
#
#     # Apply the mask to the edges
#     masked_edges = cv2.bitwise_and(edges, mask)
#     return masked_edges, poly_bottom_y, poly_top_y, mask

def detect_lanes_hough_transform(masked_edges, frame, width, center_offset):
    center_x = int(width / 2) - center_offset

    left_edges = masked_edges[:, :center_x]
    right_edges = masked_edges[:, center_x:]

    lines_left = cv2.HoughLinesP(left_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10, maxLineGap=50)

    # Detect lines on the right side of the split
    lines_right = cv2.HoughLinesP(right_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10, maxLineGap=50)

    # Adjust the x-coordinates of the detected lines on the right side
    if lines_right is not None:
        lines_right = [[[x1 - center_offset, y1, x2 - center_offset, y2]] for line in lines_right for x1, y1, x2, y2 in line]

    line_image = np.zeros_like(frame)

    return line_image, lines_left, lines_right


def filter_lines_both_sides(lines_left, lines_right, line_image, width, slope_threshold=(0.4, 4), sample_step=5):
    # Helper function to calculate line length
    def line_length(line):
        x1, y1, x2, y2 = line[0]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Helper function to extend lines
    def extend_line(line, factor=2):
        x1, y1, x2, y2 = line[0]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        x1 = mid_x + (x1 - mid_x) * factor
        y1 = mid_y + (y1 - mid_y) * factor
        x2 = mid_x + (x2 - mid_x) * factor
        y2 = mid_y + (y2 - mid_y) * factor
        return np.array([[[int(x1), int(y1), int(x2), int(y2)]]])

    # Function to filter and sort lines by length
    def filter_and_extend_lines(lines):
        filtered_lines = []
        for line in lines:
            if line_length(line) != 0:
                filtered_lines.append(line)

        # Sort lines by length in descending order and take the first 5
        filtered_lines.sort(key=line_length, reverse=True)
        longest_lines = filtered_lines[:10]

        # Extend the lines
        extended_lines = [extend_line(line) for line in longest_lines]
        return extended_lines

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

    # Process both sets of lines
    left_lines = filter_and_extend_lines(lines_left) if lines_left is not None else []
    right_lines = filter_and_extend_lines(lines_right) if lines_right is not None else []

    # Collect points from the extended lines
    left_points = [point for line in left_lines for point in filter_lines(line, 0)]
    right_points = [point for line in right_lines for point in filter_lines(line, width // 2)]

    return left_points, right_points

# def filter_lines_both_sides(lines_left, lines_right, line_image, width, slope_threshold=(0.5, 4), sample_step=5):
#     # Filter and sample the points of the left and right line segments that meet the slope requirement.
#     def filter_lines(lines, width_offset=0):
#         points = []
#         if lines is not None:
#             for line in lines:
#                 for x1, y1, x2, y2 in line:
#                     if (x2 - x1) != 0:  # 防止除以零
#                         slope = (y2 - y1) / (x2 - x1)
#                         if slope_threshold[0] < abs(slope) < slope_threshold[1]:
#                             num_points = int(np.hypot(x2 - x1, y2 - y1) // sample_step)
#                             for i in np.linspace(0, 1, num=num_points):
#                                 x = int(x1 + (x2 - x1) * i) + width_offset
#                                 y = int(y1 + (y2 - y1) * i)
#                                 points.append((x, y))
#                                 cv2.circle(line_image, (x, y), 2, (255, 255, 0), -1)
#         return points
#
#     left_points = filter_lines(lines_left)
#     right_points = filter_lines(lines_right, width // 2)
#
#     return left_points, right_points

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

# def calculate_and_visualize_lane_center(frame, left_poly, right_poly, poly_top_y, poly_bottom_y, width, step=10,
#                                         color=(255, 0, 0), thickness=2):
#
#     # Calculate the lane median, plot it onto the image, and calculate the average deviation of the median from the centre of the image.
#
#     if left_poly is None or right_poly is None:
#         return frame, 0
#
#     ys = np.arange(poly_top_y, poly_bottom_y, step)
#     left_xs = left_poly(ys)
#     right_xs = right_poly(ys)
#
#     center_xs = (left_xs + right_xs) / 2
#     lane_center_line = np.column_stack((center_xs, ys)).astype(np.int32)
#
#     # Visualisation of the lane centre line
#     for x, y in lane_center_line:
#         cv2.circle(frame, (int(x), int(y)), radius=2, color=color, thickness=thickness)
#
#     # Calculate the average of the distance from the centre of the lane to the centre of the image
#     image_center_x = width // 2
#     deviations = center_xs - image_center_x
#     average_deviation = int(np.mean(deviations))
#
#     # Text Setting
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.5
#     font_color = (255, 255, 255)
#     font_thickness = 2
#
#     # Print the coordinates and deviation at the bottom
#     cv2.line(frame, (image_center_x, poly_bottom_y), (image_center_x, poly_top_y), (0, 0, 255), 5)
#
#     cv2.putText(frame, f'Error: {average_deviation}', (image_center_x, int((poly_top_y + poly_bottom_y) / 2)), font,
#                 font_scale * 2, font_color,
#                 font_thickness, cv2.LINE_AA)
#
#     return frame

def detect_lane_lines(frame, show_steps=True):

    height, width = frame.shape[:2]

    # center_offset = int(width / 20)
    center_offset = 0

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

    masked_edges, poly_bottom_y, poly_top_y, mask = apply_mask(edges, height, width, center_offset)
    if show_steps:
        cv2.imshow('Masked Edges', masked_edges)
        cv2.imshow('Mask', mask)

    line_image, lines_left, lines_right = detect_lanes_hough_transform(masked_edges, frame, frame.shape[1], center_offset)

    # Use filter_lines function
    left_points, right_points = filter_lines_both_sides(lines_left, lines_right, line_image, frame.shape[1])

    if show_steps:
        cv2.imshow('Hough Lines', line_image)

    # Perform a polynomial fit and plot the curve while obtaining the top point
    left_poly, right_poly = fit_and_draw_polyline(left_points, right_points, line_image, height, poly_top_y, poly_bottom_y)

    if show_steps:
        cv2.imshow('Line Image', line_image)

    # # Calculate the lane centre line, plot it on the image and obtain the mean deviation
    # line_image = calculate_and_visualize_lane_center(line_image, left_poly, right_poly, poly_top_y, poly_bottom_y, width)
    #
    # if show_steps:
    #     cv2.imshow('Error Calculation', line_image)

    # Image combination
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return result


def create_next_results_folder(base_path="LaneDet_Results"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    subfolders = [f.name for f in os.scandir(base_path) if f.is_dir()]
    folder_number = 1 if not subfolders else int(
        max(subfolders, key=lambda x: int(x.replace("Result", ""))).replace("Result", "")) + 1
    new_folder = os.path.join(base_path, f"Result{folder_number}")
    os.makedirs(new_folder)
    return new_folder


def process_image(image_path):
    new_result_folder = create_next_results_folder()  # Creating a new results folder
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open/read file: {image_path}")
        return
    lane_image = detect_lane_lines(image)
    save_path = os.path.join(new_result_folder, os.path.basename(image_path))
    cv2.imwrite(save_path, lane_image)  # Save image
    print(f"Processed image saved to {save_path}")
    cv2.imshow('Final Result', lane_image)
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # If the Esc key is pressed, exit
            break
    cv2.destroyAllWindows()


def process_video(video_path):
    new_result_folder = create_next_results_folder()  # Creating a new results folder
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_path}")
        return

    # Setting video saving parameters
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(new_result_folder, 'output_video.avi'), fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lane_image = detect_lane_lines(frame)
        out.write(lane_image)  # Saving video frames
        cv2.imshow('Final Result', lane_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to {new_result_folder}")


def process_folder(folder_path):
    new_result_folder = create_next_results_folder()  # Creating a new results folder
    # Loop through all files in the folder_path
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for supported image formats
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to open/read file: {image_path}")
                continue  # Skip this file and move to the next
            lane_image = detect_lane_lines(image)
            save_path = os.path.join(new_result_folder, filename)
            cv2.imwrite(save_path, lane_image)  # Save image
            print(f"Processed image saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Lines Detection with Dynamic ROI and Preprocessing")
    parser.add_argument("--source", type=str, default='Data/Images/5.png',
                        help="Image or video file path, or camera index (e.g., 0 for the default camera)",
                        required=False)
    args = parser.parse_args()

    source = args.source

    if os.path.isdir(source):  # If source is a directory
        print(f"Processing all images in folder: {source}")
        process_folder(source)
    elif source.isdigit():  # If source is a digit, assume it's a camera index
        process_video(int(source))
    elif source.lower().endswith(('.png', '.jpg', '.jpeg')):  # If source is an image file
        process_image(source)
    else:  # Assume source is a video file
        process_video(source)
