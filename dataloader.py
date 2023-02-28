import shutil
import os
import torch
from torchvision import datasets, transforms
from PIL import Image

class generate_data_processer():
    def __init__(self,transforms=None ,PATH="../autodl-tmp/" ,TEST_DATA=100):
        '''
        find the original data in "PATH/data" and generate test data in "PATH/test"
        '''
        super(generate_data_processer,self).__init__()
        self.PATH=PATH
        self.test_data=TEST_DATA
        self.preprocess=transforms
    def generate_test_data(self):
        data_cnt = 0
        test_dic = os.path.join(self.PATH, "test")
        for person_file in os.listdir(os.path.join(self.PATH, 'data')):
            eye_file_name = os.path.join(self.PATH, 'data', person_file)
            if not os.path.isdir(eye_file_name):
                continue
            for eye_file in os.listdir(eye_file_name):
                file_dic = os.path.join(self.PATH, 'data', person_file, eye_file)
                if not os.path.isdir(file_dic):
                    continue
                for file in os.listdir(file_dic):
                    # if the data can be used
                    if not file.endswith(".jpg"):
                        continue # todo: there are some png img
                    try:
                        Image.open(os.path.join(file_dic,file))
                    except:
                        print("{} can not open".format(os.path.join(file_dic,file)))
                        continue

                    data_cnt += 1
                    if data_cnt > self.test_data:
                        return 
                    label = self.get_label(file)
                    self.push_image(test_dic, file_dic, file, label,"{}.jpg".format(str(data_cnt)))

    def push_image(self,target_dic,file_dic, file_name, label,new_name):
        target_path = os.path.join(target_dic, label)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        shutil.copy(os.path.join(file_dic,file_name), target_path)
        os.rename(os.path.join(target_path,file_name),os.path.join(target_path,new_name))
        if transforms is not None:
            img=Image.open(os.path.join(target_path,new_name))
            try :
                img=transforms(img)
            except:
                raise "generate_data_processer: transforms is not callabel"
            img.save(os.path.join(target_path,new_name))
    def get_label(file_name: str):
        '''
        task: stage the rop,
        1,2,3,4,5 as str is the stage rop
        0 no-rop
        '''
        file_str=file_name.lstrip()
        stage_list=["1","2","3","4","5"]
        if file_str.startswith("ROP"):
            # pos_cnt=pos_cnt+1
            stage=(file_str[file_str.find("期")-1])
            assert stage in stage_list,"unexpected ROP stage"
            return stage
        else:
            # neg_cnt=neg_cnt+1
            return "0"


def generate_dataloader(PATH="../autodl-tmp", train_proportion=0.6, val_proportion=0.2 ,batch_size=64, shuffle=True):
    '''
    generate train and test data in "pytorch.dataloader" format.
    '''
    # test_proportion=1-train_proportion

    data_transfrom = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    full_dataset = datasets.ImageFolder(os.path.join(
        PATH, "test"), transform=data_transfrom)  

    data_size = len(full_dataset)
    train_size = int(data_size*train_proportion)
    val_size=int(data_size*val_proportion)
    test_size = data_size-train_size-val_size

    train_dataset,val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size,val_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle)
    num_class = len(full_dataset.classes)
    return train_dataloader,val_dataloader, test_dataloader, train_size,val_size, test_size, num_class
