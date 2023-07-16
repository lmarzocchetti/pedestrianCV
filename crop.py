from PIL import Image

def crop_image(image: Image.Image, center_pos: tuple[int, int], dimension: tuple[int, int] = (512, 512)) -> tuple[Image.Image, tuple[int, int, int, int]]:
    box = [center_pos[0] - dimension[0] / 2, 
           center_pos[1] - dimension[1] / 2,
           center_pos[0] + dimension[0] / 2,
           center_pos[1] + dimension[1] / 2]
    
    if box[0] < 0:
        box[2] = box[2] + abs(box[0])
        box[0] = 0
    elif box[2] > image.width:
        box[0] = box[0] - (box[2] - image.width)
        box[2] = image.width
        
    if box[1] < 0:
        box[3] = box[3] + abs(box[1])
        box[1] = 0
    elif box[3] > image.height:
        box[1] = box[1] - (box[3] - image.height)
        box[3] = image.height

    return image.crop(tuple(box)), tuple(box)

def crop_images(images: list[Image.Image], center_pos: tuple[int, int], dimension: tuple[int, int] = (512, 512)) -> list[Image.Image]:
    return [crop_image(image, center_pos, dimension)[0] for image in images]

def crop_moving_images(images: list[Image.Image], center_pos: list[tuple[int, int]], dimension: tuple[int, int] = (512, 512)) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    return [crop_image(image, center, dimension) for (image, center) in zip(images, center_pos)]