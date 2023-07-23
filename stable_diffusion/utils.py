from PIL import Image
import numpy as np
import os

def load_frames() -> list[Image.Image]:
    l = []
    
    for filename in os.listdir('frames'):
        img = Image.open(os.path.join('frames', filename))
        num = int(filename.removeprefix("frame_").removesuffix(".jpg"))
        l.append((num,img))
    
    l.sort()
    
    return [i for (n, i) in l]

def load_masks() -> list[Image.Image]:
    l = []
    
    for filename in os.listdir('mask'):
        img = Image.open(os.path.join('mask', filename))
        num = int(filename.removeprefix("mask_").removesuffix(".jpg"))
        l.append((num,img))
    
    l.sort()
    
    return [i for (n, i) in l]

def load_keypoints() -> list[Image.Image]:
    l = []
    
    for filename in os.listdir('keypoints'):
        img = Image.open(os.path.join('keypoints', filename))
        num = int(filename.removeprefix("pose_").removesuffix(".jpg"))
        l.append((num,img))
    
    l.sort()
    
    return [i for (n, i) in l]

def substitute_rect(image: Image.Image, crop: Image.Image, start_pos: tuple[int, int]) -> Image.Image:
    ret_val = np.array(image)
    crop_ar = np.array(crop)
    
    for row in range(crop.height):
        for col in range(crop.width):
            ret_val[(start_pos[1] - 1) + row, (start_pos[0] - 1) + col] = crop_ar[row, col]
    
    return Image.fromarray(ret_val)
   
def fill_keypoints(keypoints, num):
    ret_val = []
    n_max = len(keypoints)
    i = 0
    
    while len(ret_val) != num:
        if i == n_max:
            i = 0
        ret_val.append(keypoints[i])
        i += 1
    
    return ret_val