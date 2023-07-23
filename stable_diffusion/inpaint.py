import os, gc
import argparse

import cv2
from PIL import Image
import torch
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline

import crop, utils, extract_frame, masks

def parse_rectangle_attr(to_parse: str) -> tuple[tuple[int, int], tuple[int, int]]:
    splitted = to_parse.split(',')
    
    return ((int(splitted[0]), int(splitted[1])), (int(splitted[2]), int(splitted[3])))

def reconstruct_video(output_filename: str, fps: int, scr_dim: tuple[int, int], input_frames_no) -> None:
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, scr_dim)

    l = []

    for filename in os.listdir('output'):
        img = cv2.imread(os.path.join('output', filename))
        num = int(filename.removeprefix("output").removesuffix(".jpg"))
        l.append((num,img))
        
    l.sort()

    for (_, i) in l:
        out.write(i)

    for frame in input_frames_no:
        out.write(frame)

    out.release()

def main():
    parser = argparse.ArgumentParser(description="Stable diffusion inpainting with a Keypoint controlnet")
    parser.add_argument('-m', '--mode', type=str, default='static', help="static | dynamic")
    parser.add_argument('-i', '--input', type=str, help='Input video')
    parser.add_argument('-r', '--rect', type=str, help='START_X,START_Y,WIDTH,HEIGHT')
    parser.add_argument('-n', '--number-of-inference', type=int, default=250, help='Number of inference steps')
    parser.add_argument('-s', '--strength', type=float, default=0.8, help='Strength of the denoising steps')
    parser.add_argument('-g', '--guidance-scale', type=float, default=7.5, help='Guidance scale of stable diffusion')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device: cuda | cpu | mps')
    parser.add_argument('--seed', type=int, default=185623, help='Seed of the stable diffusion generator')
    parser.add_argument('-p', '--prompt', type=str, default='Man walking in foreground, high resolution')
    parser.add_argument('--increment-x', type=int, default=0, help='If dynamic, how fast to move the mask axis x')
    parser.add_argument('--increment-y', type=int, default=0, help='If dynamic, how fast to move the mask axis y')
    parser.add_argument('--direction', type=str, default='R', help='if dynamic, direction to move the mask')
    parser.add_argument('-o', '--output', type=str, default='output.avi', help='Name of the output video')
    parser.add_argument('--enable-attention-slicing', type=str, default='n', help='y|n')
    parser.add_argument('--frames', type=int, default=None, help='For how many frames you want to inpaint?')
    args = parser.parse_args()
        
    # Create output and input directories if missing
    os.makedirs('output', exist_ok=True)
    os.makedirs('frames', exist_ok=True)
    
    # Load models
    if args.mode == 'dynamic':
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",torch_dtype= torch.float32 if args.device == 'mps' else torch.float16)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype= torch.float32 if args.device == 'mps' else torch.float16
        )
    elif args.mode == 'static':
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype= torch.float32 if args.device == 'mps' else torch.float16
        )
    else:
        print("Error in the mode argument!")
        exit(1)

    if (args.device != 'mps'):
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
    
    if (args.enable_attention_slicing == 'y'):
        pipe.enable_attention_slicing()
    
    pipe = pipe.to(args.device)

    generator_det = torch.Generator(device=args.device)
    generator_det.manual_seed(args.seed)
    initial_state = generator_det.get_state()

    prompt = args.prompt
    
    # Extract the frames from video
    # scr_dim, fps, n_frames, input_frames_all = extract_frame.read_frames(args.input)
    scr_dim, fps, n_framex = extract_frame.extract_frames(args.input)
    
    n_frames = n_framex if args.frames == None else args.frames
    
    rect_start, rect_dim = parse_rectangle_attr(args.rect)
        
    
    # Create the mask from rectangle
    if args.mode == 'static':
        mask_frames = masks.create_frame_mask(scr_dim, rect_dim, rect_start)
    else:
        mask_frames = masks.create_video_mask(scr_dim, rect_dim, rect_start, args.increment_x, args.increment_y, n_frames)
        
        print(f"mask_frames {len(mask_frames)}")
    
    # load keypoints if the mode is dynamic
    if args.mode == 'dynamic':
        keypoints = utils.load_keypoints()
    
    # control how much frames to mask
    if args.mode == 'static':
        mask_frames = [mask_frames for _ in range(n_frames)]
    else:
        input_frames = []
        for i in range(len(mask_frames)):
            input_frames.append(Image.open(f"frames/frame_{i}.jpg"))
        # input_frames = input_frames_all[:len(mask_frames)]
        # input_frames_no = input_frames_all[len(mask_frames):]


    # Creation of Cropped frames and cropped masks
    if args.mode == 'static':
        crop_frames = crop.crop_images(input_frames, (rect_start[0] + rect_dim[0]/2, rect_start[1] + rect_dim[1]/2))
        crop_mask, rect = crop.crop_image(mask_frames, (rect_start[0] + rect_dim[0]/2, rect_start[1] + rect_dim[1]/2))
        crop_masks = [crop_mask for _ in range(len(crop_frames))]
    else:
        if args.frames == None:
            moving_rect = masks.center_pos_video_mask(scr_dim, rect_dim, rect_start, args.increment_x, args.increment_y, n_frames)
        else:
            moving_rect = masks.center_pos_video_mask(scr_dim, rect_dim, rect_start, args.increment_x, args.increment_y, args.frames)
            
        moving_center = []

        for pos in moving_rect:
            moving_center.append((int(pos[0] + rect_dim[0]/2), int(pos[1] + rect_dim[1]/2)))

        crop_frames_p_rect = crop.crop_moving_images(input_frames, moving_center)
        crop_masks_p_rect = crop.crop_moving_images(mask_frames, moving_center)

        # Garbage collect
        del mask_frames
        gc.collect()

        
        crop_frames = []
        crop_masks = []
        rects = []

        for (frm, rec) in crop_frames_p_rect:
            crop_frames.append(frm)
            rects.append(rec)

        for (msk, rec) in crop_masks_p_rect:
            crop_masks.append(msk)
            
        keypoints = utils.fill_keypoints(keypoints, len(crop_frames))
    
    
    bbox_file = open("bbox.txt", "w")
    bbox_file.write("<frame> <start_x> <start_y> <width> <height>\n")

    # Generation step in base of mode
    if args.mode == 'dynamic':
        for idx, (frame, mask, keypoint) in enumerate(zip(crop_frames, crop_masks, keypoints)):
            image = pipe(prompt = prompt, image=frame,
                        mask_image=mask, height=frame.height,
                        width=frame.width,control_image=keypoint, num_inference_steps=args.number_of_inference, 
                        generator=generator_det, strength=args.strength, guidance_scale = args.guidance_scale, controlnet_conditioning_scale=0.9).images[0]

            generator_det.set_state(initial_state)
            
            new_image = utils.substitute_rect(input_frames[idx], image, (int(rects[idx][0]), int(rects[idx][1])))
            
            bbox_file.write(f"{idx} {int(rects[idx][0])} {int(rects[idx][1])} {rect_dim[0]} {rect_dim[1]}\n")
            
            new_image.save(f"output/output{idx}.jpg")
    else:
        for idx, (frame, mask) in enumerate(zip(crop_frames, crop_masks)):
            image = pipe(prompt = prompt, image=frame,
                        mask_image=mask, height=frame.height,
                        width=frame.width, num_inference_steps=args.number_of_inference, 
                        generator=generator_det, strength=args.strength, guidance_scale = args.guidance_scale).images[0]

            generator_det.set_state(initial_state)
            
            new_image = utils.substitute_rect(input_frames[idx], image, (int(rect[0]), int(rect[1])))
            
            bbox_file.write(f"{idx} {int(rect[0])} {int(rect[1])} {rect_dim[0]} {rect_dim[1]}\n")
            
            new_image.save(f"output/output{idx}.jpg")
            
    bbox_file.close()
    
    del input_frames
    del crop_frames
    del crop_masks
    del keypoints
    gc.collect()
    
    input_frames_no = [Image.open(f"frames/frame_{i}.jpg") for i in range(n_frames, n_framex)]
    
    reconstruct_video(args.output, fps, scr_dim, input_frames_no)

if __name__ == '__main__':
    main()
