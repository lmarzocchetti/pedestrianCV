# funzione che prende in input il nome del video e restituisce la lista di tutti i suoi frame
# /work/cvcs_2023_group17/Datasets/MOTSynth
import cv2

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

if __name__ == '__main__':
    frames = getFrames(0)
    print(len(frames))
    cv2.imwrite('frame2.png', frames[0])
    cv2.waitKey(0)
