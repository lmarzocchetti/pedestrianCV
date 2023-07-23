import os
f = open("../config/train.txt")

lines = f.readlines()

out = open("DAELIMINARE", "w")

for line in lines:
    image_dir = os.path.dirname(line)
    label_dir = "labels".join(image_dir.rsplit("images", 1))
    label_file = os.path.join(label_dir, os.path.basename(line))
    label_file = os.path.splitext(label_file)[0] + '.txt'
    if os.stat(label_file).st_size == 0:
        out.write(label_file + "\n")

f.close()
out.close()