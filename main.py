import torch 
from CNNs.inception_v3_pre import build_inception3_pretrained as build_model
from dataloader import generate_test_data, generate_dataloader
from train import train_process
import torch.optim as optim
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--PATH', type=str, default="../autodl-tmp/", help='Where the data is')
parser.add_argument('--TEST_DATA', type=int, default=1e10, help='How many data will be used')
parser.add_argument('--train_proportion', type=float, default=0.6, help='What proportion of the training data is')
parser.add_argument('--test_proportion', type=float, default=0.2, help='What proportion of the validation data is')
parser.add_argument('--PATH', type=str, default="../autodl-tmp/", help='where the data is')

parser.add_argument('--batch_size', type=int, default=128 ,help='batch size')
parser.add_argument('--epoch', type=int, default=10 ,help='epoch when triaing')
parser.add_argument('--lr', type=float, default=1e-3 ,help='learning rate')
parser.add_argument('--TEST_DATA', type=int, default=1e10, help='How many data will be used')
args = parser.parse_args()
loss_func=torch.nn.CrossEntropyLoss()
GEN_DATA=True

if GEN_DATA:
        generate_test_data(PATH=args.PATH, TEST_DATA=args.TEST_DATA)

train_loader, test_loader, train_len, test_len ,num_class= generate_dataloader(PATH=args.PATH,
                                                                     train_proportion=args.train_proportion, 
                                                                     batch_size=args.batch_size)
model=build_model(num_classes=num_class).cuda()#todo model_setting
train_processer=train_process(epoch=args.epoch,loss_func=loss_func)
optimizer=optim.Adam(model.parameters(), lr=args.lr)

train_processer.train(model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_len=train_len,
        test_len=test_len,
        optimizer=optimizer,
        logging=True,
        save_model=True,
        model_name="inceptionv2_pre")
        