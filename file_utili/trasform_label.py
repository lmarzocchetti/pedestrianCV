from os import path

from typing import List

class Box:
    """
        classe per ogni box di ogni frame
    """
    def __init__(self, left: int, top: int, height: int, width: int) -> None:
        self.left = left
        self.top = top
        self.height = height
        self.width = width

    def scale(self, img_width, img_height):
        self.left = (self.left + self.width/2) / img_width
        self.top = (self.top + self.height/2) / img_height
        self.width = self.width / img_width
        self.height = self.height / img_height

class FrameBoxes:
    """
        classe per ogni frame di ogni video
    """
    def __init__(self, frame: int) -> None:
        self.frame = frame
        self.boxes = []
    
    def insert(self, box: Box) -> None:
        self.boxes.append(box)

    def scale_all_frames(self, img_width, img_height):
        for i in self.boxes:
            i.scale(img_width, img_height)
    
    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index:int):
        return self.boxes[index]

def readVideo(n_video: int) -> list[FrameBoxes]:
    """
        dato il numero di un video, si ritorna una lista contenente
        altre liste, una per ogni frame del video con le cordinate
        dei vari bounding box
    """
    PATH = '/work/cvcs_2023_group17/Datasets/mot_annotations/'
    FILE = '/gt/gt.txt'
    n_video = str(n_video)
    lenght_of_input = len(n_video)
    padding_of_zero = 3-lenght_of_input
    file = PATH + '0'*padding_of_zero + n_video + FILE
    current_frame: int = 0
    retVal: list[FrameBoxes] = []

    if(path.isfile(file)):
        in_file = open(file, 'r')
        in_lines = in_file.readlines()

        for line in in_lines:
            line_splitted = line.split(',')
            frame = int(line_splitted[0])

            if frame == current_frame + 1:
                current_frame = frame
                frame_boxes = FrameBoxes(current_frame)
                retVal.append(frame_boxes)
            elif frame > current_frame: # 3 1
                missing_frames = frame - current_frame - 1
                for i in range(missing_frames):
                    retVal.append(FrameBoxes(current_frame + i + 1))
                current_frame = frame
                frame_boxes = FrameBoxes(current_frame)
                retVal.append(frame_boxes)


            frame_boxes.insert(Box(int(line_splitted[2]), int(line_splitted[3]), int(line_splitted[5]), int(line_splitted[4])))
    else:
        print('Video not exist')

    return retVal

def main():
    for video_idx in range(469):
        video_label: List[FrameBoxes] = readVideo(video_idx)

        for idx, frame in enumerate(video_label):
            frame.scale_all_frames(1920, 1080)
            with open(f"/work/cvcs_2023_group17/mot_preprocess/labels/train/{video_idx:03d}/{idx:04d}.txt", "w") as f:
                for bbox in frame:
                    f.write(f"0 {bbox.left} {bbox.top} {bbox.width} {bbox.height}\n")


if __name__ == "__main__":
    main()