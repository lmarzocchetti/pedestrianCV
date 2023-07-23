from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
#from get_frames import getFramesFromPath
from moviepy.editor import VideoFileClip

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)

# classe per caricare video da una cartella - creata da Alessandro Lorusso    
class VideoFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform
        
    def __getitem__(self, index):
        video_path = self.files[index % len(self.files)]
        clip = VideoFileClip(video_path)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            video = torch.stack([self.transform((frame, boxes))[0] for frame in clip.iter_frames()])
            print(video.shape)

        return video_path, video

    def __len__(self):
        return len(self.files)
            


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets


    def collate_fn_1(self, batch):
        # imgs is tuple, len(imgs) = sum_batchsize
        # targets is tuple, len(targets) = sum_batchsize
        paths, imgs, targets = list(zip(*batch)) # zip(*): uncompressed
                                                 # convert list to tuple
                                                 # list(): return list

        """ 
        # remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # add batch-index to targets ex: batch size 8 -> batch-index: 0, 0, 1, 2, 3 ... 7
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i 
        targets = torch.cat(targets, 0) # 0-dimension
        """ 

        # note: solve multi gpu train problem, refer to https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/181
        # find max number of targets in one image
        max_targets = 0
        for i in range(len(targets)):
            # exist no target
            if targets[i] is None:
                continue
            length_target = targets[i].size(0)
            if (max_targets < length_target):
                max_targets = length_target

        #print ('max_targets: ', max_targets)
        new_targets = []

        for i, boxes in enumerate(targets):
            if boxes is None:
                continue
            boxes[:, 0] = i
            if (boxes.size(0) < max_targets):
                append_size = max_targets - boxes.size(0)
                append_tensor = torch.zeros((append_size, 6))
                boxes = torch.cat((boxes, append_tensor), 0)
            new_targets.append(boxes)

            #print (i, boxes)
                
        #targets = [boxes for boxes in targets if boxes is not None]
        targets = [boxes for boxes in new_targets if boxes is not None]
        targets = torch.cat(targets, 0)

        # select new image size every 10 batch 
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # resize image(pad-to-square) to new size
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)