import shutil
import os
import torch
from torchvision import datasets, transforms
from PIL import Image


def generate_test_data(PATH="../autodl-tmp/", TEST_DATA=100, clear=True):
    '''
    find the original data in "PATH/data" and generate test data in "PATH/test"
    '''

    pos_cnt = 0
    neg_cnt = 0

    def push_image(target_dic,file_dic, file_name, label,new_name):
        target_path = os.path.join(target_dic, label)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        shutil.copy(os.path.join(file_dic,file_name), target_path)
        os.rename(os.path.join(target_path,file_name),os.path.join(target_path,new_name))

    def get_label(file_name: str):
        if file_name.lstrip().startswith("ROP"):
            # pos_cnt=pos_cnt+1
            return "ROP"
        else:
            # neg_cnt=neg_cnt+1
            return "NO"

    if clear:
        os.system("rm -rf {}".format(os.path.join(PATH, 'test/*')))
    data_cnt = 0
    test_dic = os.path.join(PATH, "test")
    for person_file in os.listdir(os.path.join(PATH, 'data')):
        eye_file_name = os.path.join(PATH, 'data', person_file)
        if not os.path.isdir(eye_file_name):
            continue
        for eye_file in os.listdir(eye_file_name):
            file_dic = os.path.join(PATH, 'data', person_file, eye_file)
            if not os.path.isdir(file_dic):
                continue
            for file in os.listdir(file_dic):
                if not file.endswith(".jpg"):
                    continue # todo: there are some png img
                try:
                    Image.open(os.path.join(file_dic,file))
                except:
                    print("{} can not open".format(os.path.join(file_dic,file)))
                    continue
                data_cnt += 1
                if data_cnt > TEST_DATA:
                    print("there is totally {} positive data and {} negtive data".format(
                        pos_cnt, neg_cnt))
                    return pos_cnt, neg_cnt
                label = get_label(file)
                if label == "ROP":
                    pos_cnt = pos_cnt+1
                else:
                    neg_cnt = neg_cnt+1
                push_image(test_dic, file_dic, file, label,"{}.jpg".format(str(data_cnt)))
    print("there is totally {} positive data and {} negtive data".format(
        pos_cnt, neg_cnt))
    return


def generate_dataloader(PATH="../autodl-tmp", train_proportion=0.6, batch_size=64, shuffle=True):
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
    test_size = data_size-train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle)
    num_class = len(full_dataset.classes)
    return train_dataloader, test_dataloader, len(train_dataset), len(test_dataset), num_class
