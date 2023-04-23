from os import path

class Box:
    """
        classe per ogni box di ogni frame
    """
    def __init__(self, left: int, top: int, heigth: int, width: int) -> None:
        self.left = left
        self.top = top
        self.heigth = heigth
        self.width = width

class FrameBoxes:
    """
        classe per ogni frame di ogni video
    """
    def __init__(self, frame: int) -> None:
        self.frame = frame
        self.boxes = []
    
    def insert(self, box: Box) -> None:
        self.boxes.append(box)

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

            if frame > current_frame:
                current_frame = frame
                frame_boxes = FrameBoxes(current_frame)
                retVal.append(frame_boxes)
            
            frame_boxes.insert(Box(int(line_splitted[2]), int(line_splitted[3]), int(line_splitted[4]), int(line_splitted[5])))
    else:
        print('Video not exist')

    return retVal

