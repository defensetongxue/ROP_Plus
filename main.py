import torch 
from CNNs.Inception3 import Inception3 as build_model
from dataloader import generate_test_data, generate_dataloader
from train import train_process
import torch.optim as optim

PATH = "../autodl-tmp/"
TEST_DATA = 6480
clear_original_test_data = True
train_proportion = 0.6
batch_size = 128
epoch=10
lr=1e-3
loss_func=torch.nn.CrossEntropyLoss()
generate_test_data(PATH=PATH, TEST_DATA=TEST_DATA,
                   clear=clear_original_test_data)
train_loader, test_loader, train_len, test_len ,num_class= generate_dataloader(PATH=PATH,
                                                                     train_proportion=train_proportion, 
                                                                     batch_size=batch_size)
model=build_model(num_classes=num_class)#todo model_setting
train_processer=train_process(epoch=epoch,lr=lr,loss_func=loss_func)
optimizer=optim.Adam(model.parameters(), lr=lr)

train_process.train(model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_len=train_len,
        test_len=test_len,
        logging=True,
        save_model=True)
        