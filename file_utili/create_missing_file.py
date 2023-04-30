import sys

if len(sys.argv) != 3:
    exit(1)

video = sys.argv[1]
num_frame = int(sys.argv[2])

for i in range(num_frame, 1800):
    f = open(f"/work/cvcs_2023_group17/mot_preprocess/labels/train/{video}/{i:04d}.txt", "w")
    f.close()