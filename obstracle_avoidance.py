#!/usr/bin/env python3
import time
import sys
import signal
import cv2
import numpy as np

import torch
from jetracer.nvidia_racecar import NvidiaRacecar
from torch2trt import TRTModule
from jetcam.csi_camera import CSICamera
from utils import preprocess

# Global variables
car = NvidiaRacecar()
camera = None
model_trt = TRTModule()

# Obstacle detection parameters
OBSTACLE_COLOR_LOWER = np.array([0, 100, 100])     # HSV range for obstacle color (red in this example)
OBSTACLE_COLOR_UPPER = np.array([10, 255, 255])    # Adjust these values for your obstacle color
OBSTACLE_AREA_THRESHOLD = 1000                     # Minimum pixel area to consider as obstacle

# State machine states
STATE_FOLLOWING = 0
STATE_AVOIDING = 1
current_state = STATE_FOLLOWING

# Avoidance maneuver parameters
avoid_step = 0
avoid_start_time = 0
MANEUVER_SEQUENCE = [
    {"steering": -0.8, "duration": 0.6},  # Right turn
    {"steering": 0.0, "duration": 0.8},   # Straight
    {"steering": 0.7, "duration": 0.7},   # Left turn
]

# Signal handler for safe exit
def signal_handler(sig, frame):
    print("\n[SYSTEM] Stopping the car...")
    car.throttle = 0.0
    car.steering = 0.0
    if camera and hasattr(camera, 'cap'):
        camera.cap.release()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def detect_obstacle(image):
    """Detects if there's an obstacle in the path using color thresholding"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for obstacle color
    mask = cv2.inRange(hsv, OBSTACLE_COLOR_LOWER, OBSTACLE_COLOR_UPPER)
    
    # Focus on bottom center of image (where obstacles appear first)
    height, width = mask.shape
    roi = mask[height//2:height, width//4:3*width//4]
    
    # Calculate area of detected color
    area = cv2.countNonZero(roi)
    
    # Debug: Uncomment to visualize detection
    # cv2.imshow("Obstacle ROI", roi)
    # cv2.waitKey(1)
    
    return area > OBSTACLE_AREA_THRESHOLD

def execute_avoidance_maneuver():
    """Executes the predefined avoidance sequence"""
    global avoid_step, avoid_start_time, current_state
    
    current_time = time.time()
    elapsed = current_time - avoid_start_time
    
    # Check if current maneuver step is complete
    if elapsed > MANEUVER_SEQUENCE[avoid_step]["duration"]:
        avoid_step += 1
        avoid_start_time = current_time
        
        # Check if all steps are complete
        if avoid_step >= len(MANEUVER_SEQUENCE):
            print("[AVOID] Maneuver complete! Resuming road following.")
            avoid_step = 0
            current_state = STATE_FOLLOWING
            return
    
    # Execute current step
    current_steering = MANEUVER_SEQUENCE[avoid_step]["steering"]
    car.steering = current_steering
    print(f"[AVOID] Step {avoid_step+1}/{len(MANEUVER_SEQUENCE)} | Steering: {current_steering:.2f}")

# Initialize hardware and model
def initialize():
    global camera, model_trt
    
    print("[SYSTEM] Initializing JetRacer AI Pro...")
    
    # Model loading
    try:
        print("[INFO] Loading TensorRT model...")
        model_trt.load_state_dict(torch.load('road_following_model_trt.pth'))
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    # Camera initialization
    try:
        camera = CSICamera(width=224, height=224, capture_fps=30)  # Reduced FPS for reliability
        print("[OK] Camera initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera: {e}")
        sys.exit(1)

    # Car configuration
    car.throttle = 0.16  # Base speed (adjust based on track)
    car.steering_offset = -0.05  # Mechanical calibration
    print("[INFO] System ready - starting autonomous driving")

# Main control loop
def run():
    global current_state, avoid_step, avoid_start_time
    
    # PID parameters
    Kp = 1.7
    Kd = 6.0
    last_error = 0.0

    print("Press Ctrl+C to stop")
    print("Steering range: ±1.0 | Throttle: 0.16")
    
    frame_count = 0
    start_time = time.time()

    while True:
        # Image capture
        image = camera.read()
        if image is None:
            print("[WARNING] No image from camera")
            continue
        
        # State machine
        if current_state == STATE_FOLLOWING:
            # Check for obstacles
            if detect_obstacle(image):
                print("[AVOID] Obstacle detected! Starting avoidance maneuver.")
                current_state = STATE_AVOIDING
                avoid_step = 0
                avoid_start_time = time.time()
            
            # Road following
            try:
                processed = preprocess(image).half()
                output = model_trt(processed).detach().cpu().numpy().flatten()
                
                # Model output validation
                x = float(output[0])
                x = max(-1.0, min(1.0, x))  # Clip to valid range
                
                # PD control calculation
                error = x
                derivative = error - last_error
                pid_steering = (error * Kp) + (derivative * Kd)
                
                # Steering safety clamp
                pid_steering = max(-1.0, min(1.0, pid_steering))
                
                # Apply controls
                car.steering = pid_steering
                last_error = error
                
            except Exception as e:
                print(f"[ERROR] Processing failed: {e}")
                car.throttle = 0.0
                if camera and hasattr(camera, 'cap'):
                    camera.cap.release()
                sys.exit(1)
                
        elif current_state == STATE_AVOIDING:
            execute_avoidance_maneuver()
        
        # Performance monitoring
        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)
            state_name = "FOLLOWING" if current_state == STATE_FOLLOWING else "AVOIDING"
            print(f"[STATUS] FPS: {fps:.1f} | State: {state_name} | Steering: {car.steering:.2f}")
            frame_count = 0
            start_time = time.time()

if __name__ == "__main__":
    initialize()
    try:
        run()
    except KeyboardInterrupt:
        signal_handler(None, None)