import cv2
from PIL import Image

def extract_frames(video_name: str, n_seconds: int = 10) -> tuple[tuple[int, int], int]:
    cap = cv2.VideoCapture(video_name)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
        
    frames_to_save = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"fps {fps}, frames_to_save {frames_to_save}")
    
    for i in range(frames_to_save):
        is_read, frame = cap.read()
        cv2.imwrite(f"frames/frame_{i}.jpg", frame)
        
    cap.release()
    
    return (width, height), fps, frames_to_save
    
def read_frames(video_name: str, n_seconds: int = 10) -> tuple[tuple[int, int], int, int, list[Image.Image]]:
    cap = cv2.VideoCapture(video_name)
    
    images = []
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
        
    frames_to_save = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"fps {fps}, frames_to_save {frames_to_save}")
    
    for i in range(frames_to_save):
        is_read, frame = cap.read()
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(converted))
        
    cap.release()
    
    return (width, height), fps, frames_to_save, images

def extract_frames_ram(video_name: str):
    ret_val = []
    
    cap = cv2.VideoCapture(video_name)
    
    while True:
        ret, frame = cap.read()
        if ret:
            ret_val.append(frame)
        else:
            break
    
    cap.release()
    return ret_val

def main():
    extract_frames("../Data/street.mp4")

if __name__ == "__main__":
    main()