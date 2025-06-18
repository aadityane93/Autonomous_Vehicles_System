#!/usr/bin/env python3
import time
import sys
import signal

import torch
from jetracer.nvidia_racecar import NvidiaRacecar
from torch2trt import TRTModule
from jetcam.csi_camera import CSICamera
from utils import preprocess

def signal_handler(sig, frame):
    print("\n[SYSTEM] Stopping the car...")
    car.throttle = 0.0
    car.steering = 0.0
    # Properly release the camera
    camera.cap.release()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

car = NvidiaRacecar()
print("[SYSTEM] System starts up .... Please wait")

# Model setup
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('road_following_model_trt.pth'))
print("[OK] Finish to faster inference with TensorRT")

camera = CSICamera(width=224, height=224, capture_fps=65)

# Car parameters
car.throttle = 0.16
car.steering = 0
car.steering_offset = 0

# PID parameters
STEERING_GAIN = 2
STEERING_BIAS = 0.1225

print("Successful to load parameter")
print("Press Ctrl+C to stop the car")

try:
    while True:
        image = camera.read()
        image = preprocess(image).half()
        output = model_trt(image).detach().cpu().numpy().flatten()
        x = float(output[0])
        pid_steering = x * STEERING_GAIN + STEERING_BIAS 
        print(f"[OUTPUT] AI-Output:{x:.4f} PID-Steering:{pid_steering:.4f}")
        car.steering = pid_steering
        time.sleep(0.01)

except KeyboardInterrupt:
    signal_handler(None, None)
except Exception as e:
    print(f"[ERROR] {e}")
    car.throttle = 0.0
    camera.cap.release()