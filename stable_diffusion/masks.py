import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

def create_frame_mask(scr_dim: tuple[int, int], mask_dim: tuple[int, int], 
                      start_pos: tuple[int, int]) -> Image.Image:
    scr_dim = (scr_dim[1], scr_dim[0])
    
    arr = np.zeros(scr_dim)
    
    arr[start_pos[1] : start_pos[1] + mask_dim[1], start_pos[0] : start_pos[0] + mask_dim[0]] = 255
    
    return Image.fromarray(arr).convert('RGB')

def create_video_mask(scr_dim: tuple[int, int], mask_dim: tuple[int, int], 
                      start_pos: tuple[int, int], x_inc: int = 0, y_inc: int = 0, frames: int = None) -> list[Image.Image]:
    cur_x, cur_y = start_pos
    
    ret_val = []
    
    t_frames = 0
        
    while True:
        if t_frames == frames:
            break
        # se non funziona leva dopo il maggiore uguale il mask_dim
        if (cur_x + mask_dim[0] >= scr_dim[0] - (256 - mask_dim[0]) or 
            cur_y + mask_dim[1] >= scr_dim[1] - (256 - mask_dim[1]) or
            cur_x - 256 <= 0  or 
            cur_y - 256 <= 0 
            ):
            break
                
        ret_val.append(create_frame_mask(scr_dim, mask_dim, (cur_x, cur_y)))
        
        cur_x += x_inc
        cur_y += y_inc
        
        t_frames += 1
            
    return ret_val
    
def center_pos_video_mask(scr_dim: tuple[int, int], mask_dim: tuple[int, int], 
                      start_pos: tuple[int, int], x_inc: int = 0,  y_inc: int = 0, frames: int = None) -> list[tuple[int, int]]:
    cur_x, cur_y = start_pos
    
    ret_val = []
    
    t_frames = 0
    
    while True:
        if t_frames == frames:
            break
        if (cur_x + mask_dim[0] >= scr_dim[0] - (256 - mask_dim[0]) or 
            cur_y + mask_dim[1] >= scr_dim[1] - (256 - mask_dim[1]) or 
            cur_x - 256 <= 0  or 
            cur_y - 256 <= 0 ):
            break
        
        ret_val.append((cur_x, cur_y))
        
        cur_x += x_inc
        cur_y += y_inc
        
        t_frames += 1
        
    return ret_val

def main():
    # mask_video = create_frame_mask((1920, 1080), (150, 150), (670, 880))
    mask_video = create_video_mask((1920, 1080), (128, 256), (460, 660), x_inc=10)
    
    if type(mask_video) is list:
        for idx, image in enumerate(mask_video):
            image.save(f"../Data/mask/mask_{idx}.jpg")
    else:
        mask_video.save("../Data/mask/mask.jpg")
        
if __name__ == "__main__":
    main()