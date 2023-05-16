import os
import cv2
import numpy as np
from PIL import Image
def crop(data_path, img_name, coordinate, radius=20):
    """
    Read an image from <data_path>/vessel_seg/<image_name>.jpg
    Crop the image through the coordinate, the coordinate is the center and radius is set by param
    Save the cropped image to <data_path>/crop_optic_disc/<image_name>.jpg
    """
    image_path = os.path.join(data_path, 'vessel_seg', f'{img_name}.jpg')
    image = Image.open(image_path)

    x, y = coordinate
    left = x - radius
    top = y - radius
    right = x + radius
    bottom = y + radius
    cropped_image = image.crop((left, top, right, bottom))
    
    output_path = os.path.join(data_path, 'crop_optic_disc', f'{img_name}.jpg')
    cropped_image.save(output_path)

def transforms(data_path, image_name):
    """
    Read the coordinate from <data_path>/optic_disc/<image_name>.txt with format "x y\n"
    If the optic_disc/<image_name>.txt is empty: return False
    else

    Get the original image size from <data_path>/images/<image_name>.jpg
    and target image size from <data_path>/vessel_seg/<image_name>.jpg

    Transform the coordinate in original image to where it is in target image
    Return the transformed coordinate
    """
    coordinate_file_path = os.path.join(data_path, 'optic_disc', f'{image_name}.txt')
    with open(coordinate_file_path, 'r') as f:
        content = f.read().strip()
        if not content:
            return False
        coordinate = tuple(map(int, content.split()))

    original_image_path = os.path.join(data_path, 'images', f'{image_name}.jpg')
    original_image = cv2.imread(original_image_path)
    original_height, original_width = original_image.shape[:2]

    target_image_path = os.path.join(data_path, 'vessel_seg', f'{image_name}.jpg')
    target_image = cv2.imread(target_image_path)
    target_height, target_width = target_image.shape[:2]

    x_ratio = target_width / original_width
    y_ratio = target_height / original_height

    transformed_x = int(coordinate[0] * x_ratio)
    transformed_y = int(coordinate[1] * y_ratio)

    return (transformed_x, transformed_y)

def generate_crop(data_path,radius):
    '''
    read the image in 
    '''
    image_list=os.listdir(os.path.join(data_path,'images'))
    total=0
    cnt=0
    os.makedirs(os.path.join(data_path, 'crop_optic_disc'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path, 'crop_optic_disc')}/*")
    for file_name in image_list:
        total+=1
        image_name=file_name.split('.')[0]
        transformed_cord= transforms(data_path,image_name)
        if transformed_cord:
            crop(data_path,image_name,transformed_cord,radius)
            cnt+=1
    print(f"there is {total} images with {cnt} has optic disc")

if __name__ =="__main__":
    from config import get_config
    args=get_config()
    generate_crop(args.path_tar,radius=args.crop_r)
    