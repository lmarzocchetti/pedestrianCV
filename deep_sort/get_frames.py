# funzione che prende in input il nome del video e restituisce la lista di tutti i suoi frame
# /work/cvcs_2023_group17/Datasets/MOTSynth
# modificato da Alessandro Lorusso il 12/06/2023
import cv2
import os
from moviepy.editor import VideoFileClip
from glob import iglob
import shutil
from time import time

def get_fps(video_path):
    cap = VideoFileClip(video_path)
    
    return int(cap.fps)

def get_duration(video_path):
    cap = VideoFileClip(video_path)
    
    return int(cap.duration)
        

"""
def getFrames(videoNumber):
    framesList = list()
    video = cv2.VideoCapture(f"/work/cvcs_2023_group17/Datasets/MOTSynth/{videoNumber:03d}.mp4")
    framesCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    currentFrame = 0
    while currentFrame < framesCount:
        _, frame = video.read()
        framesList.append(frame)
        currentFrame += 1
    return framesList

def getFramesFromPath(path):
    framesList = list()
    video = cv2.VideoCapture(path)
    framesCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    currentFrame = 0
    while currentFrame < framesCount:
        _, frame = video.read()
        framesList.append(frame)
        currentFrame += 1
    return framesList
"""
def write_frames(input_video_folder='deep_sort/data/input_videos', output_frames_folder='deep_sort/output_frames'):
    os.makedirs(input_video_folder, exist_ok=True)
    os.makedirs(output_frames_folder, exist_ok=True)
    videos_names = [vid for vid in sorted(iglob("%s/*.*" % input_video_folder)) if vid.endswith(".mp4")]
    if len(videos_names) != 0:
        os.makedirs(output_frames_folder, exist_ok=True)    
        for vid in videos_names:
            video_id = os.path.basename(vid).split(".")[0]
            frames_path = f'{output_frames_folder}/{video_id}'
            shutil.rmtree(frames_path, ignore_errors=True)
            os.makedirs(frames_path, exist_ok=True)
            vid_capture = VideoFileClip(vid)

            vid_capture.write_images_sequence(f'{frames_path}/%04d.jpg')
    else:
        print("No videos in input folder")
"""
if __name__ == '__main__':
    os.chdir("/work/cvcs_2023_group17/mot_preprocess/train/")
    video_index = 0
    frame_index = 0
    while video_index < 718:
        if not os.path.exists(f"{video_index:03d}"):
            os.mkdir(f"{video_index:03d}")
            print(f"Video {video_index} folder created.")
        frames = getFrames(video_index)
        while frame_index < len(frames):
            if not os.path.exists(f"{video_index:03d}/{frame_index:04d}.jpg"):
                cv2.imwrite(f"{video_index:03d}/{frame_index:04d}.jpg", frames[frame_index])
                print(f"Video {video_index} Frame {frame_index} extracted")
            frame_index += 1
        video_index += 1
"""
if __name__ == '__main__':
    print(get_duration("data/input_videos/765.mp4"))
