
from VesselSegModule import VesselSegProcesser
import os 
def generate_vessel(PATH):
    '''generate vessel segmentation and save
    this file will search the data in an "ImageFolder" class 
    located at Path/test and output the segmentation results
    to PATH/vessel_res. Please note that files in the image folder
    cannot have the same name in different classes. 
    For example, the following example would be considered invalid
    ...
    PATH/test/cat/kitty.jpg
    ...
    PATH/tes/dog/kitty.png
    ...
    In this case, the file name "kitty" has been used for both 
    the cat and dog classes, which is not allowed.
    
    '''
    vessel_seg_processr=VesselSegProcesser(model_name='FR_UNet',
                                            save_path=os.path.join(PATH,'vessel_res'),
                                            resize=(512,512))
    image_folder_dic=os.path.join(PATH,'test')
    for classes in os.listdir(image_folder_dic):
        class_dic=os.path.join(image_folder_dic,classes)
        for file in os.listdir(class_dic):
            vessel_seg_processr(img_path=os.path.join(class_dic,file))
    return 
if __name__=="__main__":
    generate_vessel('../autodl-tmp/mini_dataset')