import os
import cv2
import glob


lane = "left_lane_data"


DATASET_DIR = f"{lane}/apex"  # adjust path if needed
deleted_log = "deleted_images.txt"
count = 0

with open(deleted_log, "w") as log:
    for img_path in glob.glob(os.path.join(DATASET_DIR, "*.jpg")):
        img = cv2.imread(img_path)
        if img is None:
            try:
                os.remove(img_path)
                log.write(f"{img_path}\n")
                print(f"[REMOVED] {img_path}")
                count += 1
            except Exception as e:
                print(f"[ERROR] Could not delete {img_path}: {e}")

print(f"\n Clean complete. Removed {count} unreadable images.")
