import os
import os.path

output_file_path = "../config/val.txt"

f_out = open(output_file_path, "w")

for video_idx in range(470, 474):
    for img_idx in range(1800):
        if os.path.isfile(f"/work/cvcs_2023_group17/mot_preprocess/labels/val/{video_idx:03d}/{img_idx:04d}.txt") and os.stat(f"/work/cvcs_2023_group17/mot_preprocess/labels/val/{video_idx:03d}/{img_idx:04d}.txt").st_size != 0:
            f_out.write(f"/work/cvcs_2023_group17/mot_preprocess/images/val/{video_idx:03d}/{img_idx:04d}.jpg\n")

f_out.close()