from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from utils_ import ROP_Dataset
import pickle 
import numpy as np
from .function_ import get_instance
from utils_ import preprocess_hander

class generate_data_processer():
    def __init__(self,PATH="../autodl-tmp/",data_file='orignal'):
        '''
        find the original data in "PATH/data" and generate test data in "PATH/{data_file}"
        '''
        super(generate_data_processer,self).__init__()
        self.PATH=PATH
        self.data_file=os.path.join(self.PATH,data_file)
        self.preprocess=get_instance(preprocess_hander,data_file)
    def generate_test_data(self):
        print("generate data Beginning...")
        data_cnt = 0
        # remove the exit test data in PATH
        if not os.path.exists(self.data_file):
            os.mkdir(self.data_file)
        os.system("rm -rf {}".format(os.path.join(self.data_file, "*")))

        for person_file in os.listdir(os.path.join(self.PATH, 'data')):
            eye_file_name = os.path.join(self.PATH, 'data', person_file)
            person_state=False # False: normal True: ROP 
            act_squeence=[]
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
                    label = self.get_label(file,file_dic)

                    if label==-1:
                        continue
                    if not label=="0": # that person is a ROP infants
                        person_state=True
                        # push that ROP image as positive sample
                        self.push_image(
                            self.data_file,image,
                            label,"{}.pkl=".format(str(data_cnt)))
                    else:
                        # else push the act into a list
                        # if the infant is non-ROP 
                        # push all the images as negtive samples
                        act_squeence.append(
                            (self.data_file,image,
                            label,"{}.pkl=".format(str(data_cnt))))

                    # self.push_image(self.data_file,image, label,"{}.pkl=".format(str(data_cnt)))
            if not person_state:
                # if there is no ROP in the infants
                for args in act_squeence:
                    self.push_image(*args)
    def push_image(self,target_dic,img, label,new_name):
        target_path = os.path.join(target_dic, label)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
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
            if stage=='p': # 没有期字
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
        for sub_class in os.listdir(os.path.join(self.data_file)):
            sub_class_number=len(os.listdir(os.path.join(self.data_file,sub_class)))
            print("{} : {}".format(sub_class,sub_class_number))

def generate_dataloader(PATH="../autodl-tmp/orignal"
                        , train_proportion=0.6, val_proportion=0.2 ,
                        batch_size=64,
                        shuffle=True):
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
    args = parser.parse_args()
    data_processer=generate_data_processer(PATH=args.PATH)
    data_processer.generate_test_data()