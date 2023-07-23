from glob import iglob
import cv2
from moviepy.editor import ImageClip, concatenate_videoclips

videos_names = [ImageClip(cv2.imread(vid, cv2.IMREAD_COLOR)).set_duration(0.05) for vid in sorted(iglob("test_frames/*"))]

concat_clip = concatenate_videoclips(videos_names, method="chain")
concat_clip.write_videofile("test.mp4", fps=20)
