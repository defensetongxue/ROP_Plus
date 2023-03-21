from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from utils import ROP_Dataset
import pickle 
import numpy as np
from .preprocess_hander import orignal_processer as preprocesser

    

class generate_data_processer():
    def __init__(self,PATH="../autodl-tmp/",data_file='orignal' ,TEST_DATA=100):
        '''
        find the original data in "PATH/data" and generate test data in "PATH/{data_file}"
        '''
        super(generate_data_processer,self).__init__()
        self.PATH=PATH
        self.test_data=TEST_DATA
        self.data_file=os.path.join(self.PATH,data_file)
        self.preprocess=preprocesser()
    def generate_test_data(self):
        print("generate data Beginning...")
        data_cnt = 0
        # remove the exit test data in PATH
        if not os.path.exists(self.data_file):
            os.mkdir(self.data_file)
        os.system("rm -rf {}".format(os.path.join(self.data_file, "*")))

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
                        image=Image.open(os.path.join(file_dic,file))
                    except:
                        print("{} can not open".format(os.path.join(file_dic,file)))
                        continue
                    # generate vessel and saved
                    image=self.preprocess(image)
                    data_cnt += 1
                    if data_cnt > self.test_data:
                        return 
                    label = self.get_label(file,file_dic)
                    if label==-1:
                        continue
                    self.push_image(self.data_file,image, label,"{}.pkl=".format(str(data_cnt)))

    def push_image(self,target_dic,img, label,new_name):
        target_path = os.path.join(target_dic, label)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        # cv2.imwrite(os.path.join(target_path,new_name),img)
        with open(file=os.path.join(target_path,new_name),mode='wb') as file:
            pickle.dump(np.array(img), file)
    def get_label(self,file_name: str,file_dir:str):
        '''
        task: stage the rop,
        1,2,3,4,5 as str is the stage rop
        0 no-rop
        6: 消退期
        -1 待排
        '''
        file_str=file_name.replace(" ","")

        stage_list=["1","2","3","4","5","行","退"]
        if file_str.startswith("ROP"):
            # pos_cnt=pos_cnt+1
            stage=(file_str[file_str.find("期")-1])
            if stage=='p':
                print(os.path.join(file_dir,file_name))
                return -1
            assert stage in stage_list,"unexpected ROP stage : {} in file {}".format(stage,file_str)
            if stage=="行" or stage=="退":
                return "6"
            return stage
        else:
            # neg_cnt=neg_cnt+1
            return "0"
    def get_data_condition(self):
        for sub_class in os.listdir(os.path.join(self.PATH,"test")):
            sub_class_number=len(os.listdir(os.path.join(self.PATH,'test',sub_class)))
            print("{} : {}".format(sub_class,sub_class_number))

def generate_dataloader(PATH="../autodl-tmp/orignal", train_proportion=0.6, val_proportion=0.2 ,batch_size=64, shuffle=True):
    '''
    generate train and test data in "pytorch.dataloader" format.
    '''
    # test_proportion=1-train_proportion

    full_dataset = ROP_Dataset(PATH)  
    
    num_class = len(full_dataset.class_dic)
    data_size = len(full_dataset)
    train_size = int(data_size*train_proportion)
    val_size=int(data_size*val_proportion)
    test_size = data_size-train_size-val_size

    train_dataset,val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size,val_size, test_size])
    val_dataset.mode=test_dataset.mode='val'
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle)
    print("the dataset has the classes: {}".format(full_dataset.classes))
    return (train_dataloader,val_dataloader, test_dataloader), (train_size,val_size, test_size), num_class

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--PATH', type=str, default="../autodl-tmp/", help='Where the data is')
    parser.add_argument('--TEST_DATA', type=int, default=1e10, help='How many data will be used')
    args = parser.parse_args()
    data_processer=generate_data_processer(PATH=args.PATH, TEST_DATA=args.TEST_DATA)
    data_processer.generate_test_data()