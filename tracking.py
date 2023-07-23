import sys
import cv2

sys.path.insert(0, "two_stream")

from two_stream.recognize import rec

from deep_sort.track_dict_people import run

from argparse import ArgumentParser
import gc
import os

def print_video(input_video, output_video, dict_action, video_dict):
    names = os.listdir(f"deep_sort/output_frames/{input_video}")
    frames = {f"deep_sort/output_frames/{input_video}/{i}": cv2.imread(f"deep_sort/output_frames/{input_video}/{i}") for i in names}
    
    for id, action in dict_action.items():
        bboxes = video_dict[int(id)]
        for path, bbox in bboxes.items():
            cv2.rectangle(frames[path], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (36,255,12), 1)
            cv2.putText(frames[path], f"id:{id}, action:{action}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    sorted_frames = sorted(frames.items(), key=lambda x: int(x[0].split("/")[3].removesuffix(".jpg")))
    
    cap = cv2.VideoCapture(f"deep_sort/data/input_videos/{input_video}.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    out = cv2.VideoWriter(f"{output_video}", cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

    for (_, frame) in sorted_frames:
        out.write(frame)
    
    out.release()
    
    
def main():
    parser = ArgumentParser(description="Tracking + Action Recognition")
    parser.add_argument("-i", "--input_name", type=str, help="Name of input video", required=True)
    parser.add_argument("--output_video", type=str, default="output.mp4", help="Path to input video")
    parser.add_argument("-m", "--model", type=str, default="deep_sort/config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="deep_sort/checkpoints/yolov3_ckpt_part4_250.pth", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-v", "--videos", type=str, default="deep_sort/data/input_videos", help="Path to directory with frames to inference")
    parser.add_argument("-c", "--classes", type=str, default="deep_sort/data/mot.names", help="Path to classes label file (.names)")
    parser.add_argument("-o", "--output", type=str, default="deep_sort/output_detected_videos", help="Path to output directory for detected videos")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=2, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.55, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="IOU threshold for non-maximum suppression")
    parser.add_argument("--n_frames", type=float, default=-1, help="Number of frames to detect and stick together")
    parser.add_argument("--write_frames", type=bool, default=True, help="Write all frames to folder")
    parser.add_argument("--frames_folder", type=str, default="deep_sort/output_frames", help="Path where to save frames, if write_frames flag is set")
    args = parser.parse_args()
    
    video_dict = run(args)
    
    gc.collect()
    
    dict_action = rec(args.input_name)
    
    gc.collect()
    
    print(video_dict)
    print("----------------------------------------------------\n")
    print(dict_action)
    
    #print_video(args.input_name, args.output_video, dict_action, video_dict)
    

if __name__ == "__main__":
    main()