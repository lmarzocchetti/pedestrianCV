from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np
import cv2
import os

from extract_frame import extract_frames_ram

def extract_pose(video_path: str, mask_dim: tuple[int, int], device, mask_start: tuple[int, int] = None):
    frames = extract_frames_ram(video_path)
    poses = []
    boxes = []
    
    mask_start_x = int(256 - mask_dim[0] / 2)
    mask_start_y = int(256 - mask_dim[1] / 2)
    
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    openpose.to(device)
    
    for frame in frames:
        poses.append(np.asarray(openpose(frame, include_hand=True, include_face=True)))
        
    for pose in poses:
        gry = cv2.cvtColor(pose, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gry, (3, 3), 0)
        th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(th)
        x, y, w, h = cv2.boundingRect(coords)
        
        crop = np.asarray(Image.fromarray(pose[y : y + h, x : x + w, :]).resize(mask_dim))
        ret = np.zeros((512, 512, 3))
        # ret[mask_start[0] : mask_start[0] + mask_dim[1], mask_start[1] : mask_start[1] + mask_dim[0], :] = crop
        ret[mask_start_y : mask_start_y + mask_dim[1], mask_start_x : mask_start_x + mask_dim[0], :] = crop
        boxes.append(ret)
        
    
    for idx, pose in enumerate(boxes):
        cv2.imwrite(f"keypoints/pose_{idx}.jpg", pose)
    
def main():
    os.makedirs('keypoints', exist_ok=True)
    extract_pose("Walking_person.mp4", (128, 256), (0, 0))
    
if __name__ == "__main__":
    main()