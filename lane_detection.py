# -*- coding: utf-8 -*-


import cv2
import numpy as np

def process_frame(image):
    # Convert the input image to HLS color space
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Define color range for white mask
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image_hls, lower_threshold, upper_threshold)

    # Define color range for yellow mask
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(image_hls, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply the mask to the input image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert the masked image to grayscale
    masked_image_gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to the grayscale image
    masked_image_gray_blur = cv2.GaussianBlur(masked_image_gray, (13, 13), 0)

    # Apply Canny edge detection to the blurred image
    masked_image_gray_blur_edge_detec = cv2.Canny(masked_image_gray_blur, 50, 150)

    # Create a region of interest mask
    mask = np.zeros_like(masked_image_gray_blur_edge_detec)
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Apply the region of interest mask to the edge detected image
    masked_image = cv2.bitwise_and(masked_image_gray_blur_edge_detec, mask)

    # Apply Hough transform to the masked image
    hough_lines = cv2.HoughLinesP(masked_image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

    # Draw Hough lines on a copy of the input image
    image_co = np.copy(image)
    for line in hough_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image_co, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Separate left and right lane lines
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in hough_lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))  # Calculate the length of the line
            if slope < 0:  # If the slope is negative, it's a left line
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:  # If the slope is positive, it's a right line
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # Calculate the weighted average of the left and right lane lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    # Function to convert a line's slope and intercept to pixel points
    def pixel_points(y1, y2, line):
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)

        return ((x1, y1), (x2, y2))

    # Calculate pixel points for left and right lane lines
    y1 = image.shape[0]
    y2 = y1 * 0.65
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)

    # Draw left and right lane lines on a blank image
    line_image = np.zeros_like(image)
    for line in (left_line, right_line):
        if line is not None:
            cv2.line(line_image, *line, [0, 255, 0], 5)

    # Combine the input image and the line image
    draw_lane_lines = cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    # If left and right lane lines are detected, add a green overlay between them
    if left_line is not None and right_line is not None:
        green_overlay = np.zeros_like(image)
        pts = np.array([left_line[0], left_line[1], right_line[1], right_line[0]], dtype=np.int32)
        cv2.fillPoly(green_overlay, [pts], (0, 255, 0))
        line_image = cv2.addWeighted(line_image, 1, green_overlay, 0.3, 0)

    # Combine the input image and the final line image (with green overlay if applicable)
    draw_lane_line = cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    return draw_lane_line

import os
from moviepy.editor import VideoFileClip

# Function to process video frames using the `process_frame` function and save the result
def process_video(input_video_path, output_video_path):
    # Read the input video
    input_video = VideoFileClip(input_video_path)
    # Apply the `process_frame` function to each frame of the input video
    processed_video = input_video.fl_image(process_frame)
    # Write the processed video to the output path without audio
    processed_video.write_videofile(output_video_path, audio=False)

# Define input and output folder paths
input_folder = 'test_videos'
output_folder = 'output_videos'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all video files in the input folder
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)

    # Check if the file is a video file
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        input_video_path = file_path

        # Replace the input folder path with the output folder path in the file name
        output_video_path = os.path.join(output_folder, file)

        # Process the video and save the output
        process_video(input_video_path, output_video_path)

import os
import cv2
import numpy as np
from glob import glob

# Function to process images using the `process_frame` function and save the result
def process_image(input_image_path, output_image_path):
    # Read the input image
    input_image = plt.imread(input_image_path)
    # Process the image using the `process_frame` function
    processed_image = process_frame(input_image)
    # Write the processed image to the output path
    cv2.imwrite(output_image_path, processed_image)

# Define input and output folder paths
input_folder = 'input_images'
output_folder = 'output_images'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all image files in the input folder
for file in glob(os.path.join(input_folder, '*')):
    # Check if the file is an image file
    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        input_image_path = file
        # Replace the input folder path with the output folder path in the file name
        output_image_path = os.path.join(output_folder, os.path.basename(file))
        # Process the image and save the output
        process_image(input_image_path, output_image_path)

