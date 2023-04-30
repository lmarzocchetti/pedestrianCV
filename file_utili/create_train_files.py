import random
import os
import os.path

output_file_path = "../config/train.txt"

f_out = open(output_file_path, "w")

for video_idx in range(469):
    for img_idx in range(33):
        num_rand = random.randint(0, 1799)
        if os.path.isfile(f"/work/cvcs_2023_group17/mot_preprocess/labels/train/{video_idx:03d}/{num_rand:04d}.txt") and os.stat(f"/work/cvcs_2023_group17/mot_preprocess/labels/train/{video_idx:03d}/{num_rand:04d}.txt").st_size != 0:
            f_out.write(f"/work/cvcs_2023_group17/mot_preprocess/images/train/{video_idx:03d}/{num_rand:04d}.jpg\n")

f_out.close()
