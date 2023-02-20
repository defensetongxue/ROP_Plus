import torch 
from CNNs.inception_v3_pre import build_inception3_pretrained as build_model
from dataloader import generate_test_data, generate_dataloader
from train import train_process
import torch.optim as optim

PATH = "../autodl-tmp/"
TEST_DATA = 1e10
clear_original_test_data = True
train_proportion = 0.6
batch_size = 128
epoch=10
lr=1e-3
loss_func=torch.nn.CrossEntropyLoss()
GEN_DATA=True

if GEN_DATA:
        generate_test_data(PATH=PATH, TEST_DATA=TEST_DATA,
                   clear=clear_original_test_data)

train_loader, test_loader, train_len, test_len ,num_class= generate_dataloader(PATH=PATH,
                                                                     train_proportion=train_proportion, 
                                                                     batch_size=batch_size)
model=build_model(num_classes=num_class).cuda()#todo model_setting
train_processer=train_process(epoch=epoch,lr=lr,loss_func=loss_func)
optimizer=optim.Adam(model.parameters(), lr=lr)

train_processer.train(model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_len=train_len,
        test_len=test_len,
        optimizer=optimizer,
        logging=True,
        save_model=True,
        model_name="inceptionv2_pre")
        