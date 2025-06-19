#!/usr/bin/env python3
import time
import sys
import signal

import torch
from jetracer.nvidia_racecar import NvidiaRacecar
from torch2trt import TRTModule
from jetcam.csi_camera import CSICamera
from utils import preprocess

# Global variables
car = NvidiaRacecar()
camera = None
model_trt = TRTModule()

# Signal handler for safe exit
def signal_handler(sig, frame):
    print("\n[SYSTEM] Stopping the car...")
    car.throttle = 0.0
    car.steering = 0.0
    if camera and hasattr(camera, 'cap'):
        camera.cap.release()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Initialize hardware and model
def initialize():
    global camera, model_trt
    
    print("[SYSTEM] Initializing JetRacer AI Pro...")
    
    # Model loading with error handling
    try:
        print("[INFO] Loading TensorRT model...")
        model_trt.load_state_dict(torch.load('220road_following_model_trt.pth'))
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    # Camera initialization
    try:
        camera = CSICamera(width=224, height=224, capture_fps=65)
        print("[OK] Camera initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera: {e}")
        sys.exit(1)

    # Car configuration
    car.throttle = 0.17  # Base speed (adjust based on track)
    car.steering_offset = -0.00  # Mechanical calibration
    print("[INFO] System ready - starting autonomous driving")

# Main control loop
def run():
    # PID parameters
    Kp = 1.7
    Kd = 6
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
        
        # Preprocess and inference
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
            
            # Performance monitoring
            frame_count += 1
            if frame_count % 60 == 0:
                fps = frame_count / (time.time() - start_time)
                print(f"[INFO] Current FPS: {fps:.1f} | Steering: {pid_steering:.2f}")
                frame_count = 0
                start_time = time.time()
                
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            car.throttle = 0.0
            camera.cap.release()
            sys.exit(1)

if __name__ == "__main__":
    initialize()
    try:
        run()
    except KeyboardInterrupt:
        signal_handler(None, None)