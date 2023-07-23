#! /usr/bin/env python3
from __future__ import division

from os import makedirs
from os.path import basename, join
from argparse import ArgumentParser
from tqdm import tqdm
from random import sample
from numpy import zeros, array, linspace, where, fromstring, uint8

from PIL import Image

import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import load_model
from utils import load_classes, rescale_boxes, non_max_suppression
from datasets import ImageFolder
from transformation import Resize, DEFAULT_TRANSFORMS

from matplotlib.pyplot import figure, subplots, get_cmap, text, axis, gca, savefig, close, draw, tight_layout, subplots_adjust, autoscale, margins
#import matplotlib.colors as colors_plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from glob import iglob
from moviepy.editor import ImageClip, concatenate_videoclips
from cv2 import imwrite, cvtColor, COLOR_RGB2BGR
from sys import exit
from get_frames import write_frames
from time import time

# Importing DeepSort
from deep_sort.deep_sort import DeepSort

# function that initializes DeepSort tracker
def get_deepsort(path_to_ckpt):
    return DeepSort(path_to_ckpt)

def detect_directory(model, weights_path, img_path, classes, output_path, n_frames, tracker,
                     batch_size=8, img_size=416, n_cpu=2, conf_thres=0.5, nms_thres=0.5):
    """
    Detects objects on all images in specified directory and saves output images with drawn detections.
    
    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to directory with images to inference
    :type img_path: str
    :param classes: List of class names
    :type classes: [str]
    :param output_path: Path to output directory
    :type output_path: str
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    """
    
    dataloader = _create_data_loader(img_path, batch_size, img_size, n_cpu)
    #model = load_model(model_path, weights_path)
    img_detections, imgs = detect(
        model,
        dataloader,
        output_path,
        conf_thres,
        nms_thres)
    _draw_and_save_detected_video(
        img_detections, imgs, tracker, img_size, output_path, classes, n_frames)

    print(f"---- Detections were saved to: '{output_path}' ----")

            
        
    

def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (image, zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections.numpy()

def detect(model, dataloader, output_path, conf_thres, nms_thres):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
    # Create output directory, if missing
    #makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    for (img_paths, input_imgs) in tqdm(dataloader, desc="Detecting"):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
        img_detections.extend(detections)
        imgs.extend(img_paths)
    return img_detections, imgs



def video_inference(model_path, weights_path, videos_path, classes, frames_path, output_videos_path, n_frames, write_frames_ok, tracker,
                     batch_size=8, img_size=416, n_cpu=2, conf_thres=0.5, nms_thres=0.5):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
    
    # If flag "write_frames is true, writes all frames to a predefined folder"
    if write_frames_ok:
        write_frames(videos_path, frames_path)
    
    # Create output directory for videos
    makedirs(output_videos_path, exist_ok=True)
    
    
    # Get all folders in input path
    folders_names = [vid for vid in sorted(iglob("%s/*" % frames_path))]
    
    #video_detections = [] # Stores detections for each video
    # Load the model
    model = load_model(model_path, weights_path)
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    folders_processed = 0
    # Iterate through all videos
    while folders_processed < len(folders_names):
        video_id = folders_names[folders_processed].split("/")[1]
        out_id_path = f'{output_videos_path}/{video_id}.mp4'
        #makedirs(out_id_path, exist_ok=True)
        detect_directory(
            model,
            weights_path,
            folders_names[folders_processed],
            classes,
            out_id_path,
            n_frames,
            tracker,
            batch_size,
            img_size,
            n_cpu,
            conf_thres,
            nms_thres)
        
        folders_processed += 1

        #video_detections.extend((img_detections, imgs))    
    
    #return video_detections


def _draw_and_save_detected_video(img_detections, imgs, tracker, img_size, output, classes, n_frames):
    """Draws detections in output images and stores them.

    :param img_detections: List of detections
    :type img_detections: [Tensor]
    :param imgs: List of paths to image files
    :type imgs: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    detected_frames = []
    i = 0

    # Iterate through images and save plot of detections
    for (image_path, detections) in tqdm(zip(imgs, img_detections), desc="Draw detections and compose tracked clip..."):
        if i == n_frames:
            break
        #print(f"Image {image_path}:")
        detected_frame = _draw_detected_frame(
            image_path, detections, tracker, img_size, classes)
        detected_frames.extend([detected_frame])
        i += 1
    
    concat_clip = concatenate_videoclips(detected_frames, method="chain")
    concat_clip.write_videofile(output, fps=20)

def bbox_rel(bbox_left, bbox_top, bbox_w, bbox_h):
    """" Calculates the relative bounding box from absolute pixel values. """
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def _draw_detected_frame(image_path, detections, tracker, img_size, classes):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # Create plot
    im = Image.open(image_path)
    dpi = 300
    img = array(im)
    figure()
    fig, ax = subplots(1)
    ax.imshow(img)
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = get_cmap("tab20b")
    colors = [cmap(i) for i in linspace(0, 1, n_cls_preds)]
    bbox_colors = sample(colors, n_cls_preds)
    
    bbox_xywh = []
    confs = []
    
    for x1, y1, x2, y2, conf, cls_pred in detections:

        #print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        # box_h = x2 - x1
        # box_w = y2 - y1
        
        bbox_left = min(x1, x2)
        bbox_top = min(y1, y2)

        bbox_w = abs(x1 - x2)
        bbox_h = abs(y1 - y2)
        
        x_c, y_c, bbox_w, bbox_h = bbox_rel(bbox_left, bbox_top, bbox_w, bbox_h)
        
        # Puts transformed coordinates and confidence into lists
        bbox_xywh.append([x_c, y_c, bbox_w, bbox_h])
        confs.append(conf)
        
        # Updates deepsort
        outputs = tracker.update((torch.Tensor(bbox_xywh)), (torch.Tensor(confs)), img)
        
        if (len(outputs) > 0):
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            color = bbox_colors[int(where(unique_labels == int(cls_pred))[0])]
            draw_boxes(ax, bbox_xyxy, color, identities)

    # Save generated image with detections
    axis("off")
    gca().xaxis.set_major_locator(NullLocator())
    gca().yaxis.set_major_locator(NullLocator())
    
    # Adjust image to remove white margin
    fig.set_dpi(dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    w,h = fig.get_size_inches()
    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    fig.set_size_inches(w, ax.get_aspect()*(y1-y2)/(x2-x1)*w)
    fig.canvas.draw()
    
    # Convert plot to an RGB numpy array
    detected_img = fromstring(fig.canvas.tostring_rgb(), dtype=uint8, sep='')
    detected_img  = detected_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #detected_img = cvtColor(detected_img, COLOR_RGB2BGR)
    #imwrite("test_frame.jpg", detected_img)
    close()
    
    return ImageClip(detected_img).set_duration(0.05)

def draw_boxes(ax, bbox, bbox_color, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        box_w = abs(x1 - x2)
        box_h = abs(y1 - y2)
        # box text and bar
        _id = int(identities[i]) if identities is not None else 0    
        # Create a Rectangle patch
        bbox = Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=[0, 1, 0, 1], facecolor="none")

        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        text(
            x1,
            y1,
            s=f"id : {_id}",
            color="white",
            verticalalignment="bottom",
            bbox={"color": bbox_color, "pad": 0},
            fontsize='xx-small')

def _create_data_loader(img_path, batch_size, img_size, n_cpu):
    """Creates a DataLoader for inferencing.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ImageFolder(
        img_path,
        transform=Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader


def run():
    parser = ArgumentParser(description="Detect objects on images.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="checkpoints/yolov3_ckpt_part4_250.pth", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-v", "--videos", type=str, default="data/input_videos", help="Path to directory with frames to inference")
    parser.add_argument("-c", "--classes", type=str, default="data/mot.names", help="Path to classes label file (.names)")
    parser.add_argument("-o", "--output", type=str, default="output_detected_videos", help="Path to output directory for detected videos")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=2, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.55, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="IOU threshold for non-maximum suppression")
    parser.add_argument("--n_frames", type=float, default=-1, help="Number of frames to detect and stick together")
    parser.add_argument("--write_frames", type=bool, default=False, help="Write all frames to folder")
    parser.add_argument("--frames_folder", type=str, default="output_frames", help="Path where to save frames, if write_frames flag is set")
    args = parser.parse_args()

    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names
    
    # Initialize deepsort
    deepsort = get_deepsort("deep_sort/deep/checkpoint/ckpt.t7")
    
    print(deepsort)
    

    video_inference(
        args.model,
        args.weights,
        args.videos,
        classes,
        args.frames_folder,
        args.output,
        args.n_frames,
        args.write_frames,
        deepsort,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres)


if __name__ == '__main__':
    run()