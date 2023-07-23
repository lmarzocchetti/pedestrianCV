# funzione che prende in input il nome del video e restituisce la lista di tutti i suoi frame
# /work/cvcs_2023_group17/Datasets/MOTSynth
import cv2
import os
from moviepy.editor import VideoFileClip

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
# 516, 528
if __name__ == '__main__':
    os.chdir("/work/cvcs_2023_group17/Datasets/Detection/")
    video_indexes = [767]
    for video_index in video_indexes:
        if not os.path.exists(f"{video_index:03d}"):
            os.mkdir(f"{video_index:03d}")
            print(f"Video {video_index} folder created.")
        video = VideoFileClip(f"/work/cvcs_2023_group17/Datasets/MOTSynth/{video_index:03d}.mp4")
        video.write_images_sequence(f'{video_index:03d}/%04d.jpg')
        video_index += 1
