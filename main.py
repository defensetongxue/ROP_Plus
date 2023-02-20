import torch 
from CNNs.inception_v3 import build_inception3_pretrained as build_model
from dataloader import generate_test_data, generate_dataloader
from train import train_process
import torch.optim as optim
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--PATH', type=str, default="../autodl-tmp/", help='Where the data is')
parser.add_argument('--TEST_DATA', type=int, default=1e10, help='How many data will be used')
parser.add_argument('--train_proportion', type=float, default=0.6, help='What proportion of the training data is')
parser.add_argument('--val_proportion', type=float, default=0.2, help='What proportion of the validation data is')

parser.add_argument('--batch_size', type=int, default=128 ,help='batch size')
parser.add_argument('--epoch', type=int, default=15 ,help='epoch when triaing')
parser.add_argument('--lr', type=float, default=1e-3 ,help='learning rate')
parser.add_argument('--GEN_DATA', type=bool, default=False, help='if generate data again')
parser.add_argument('--pretrain', type=bool, default=True, help='if load the pretrain model')
args = parser.parse_args()
print("Begin pretrain {}".format(args.pretrain))
loss_func=torch.nn.CrossEntropyLoss()
if args.GEN_DATA:
        generate_test_data(PATH=args.PATH, TEST_DATA=args.TEST_DATA)

train_loader, val_loader,test_loader, train_len, val_len,test_len ,num_class= generate_dataloader(PATH=args.PATH,
                                                                     train_proportion=args.train_proportion, 
                                                                     val_proportion=args.val_proportion,
                                                                     batch_size=args.batch_size)
model=build_model(num_classes=num_class,pretrained=args.pretrain).cuda()#todo model_setting
train_processer=train_process(epoch=args.epoch,loss_func=loss_func)
optimizer=optim.Adam(model.parameters(), lr=args.lr)

train_processer.train(model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_len=train_len,
        val_len=val_len,
        test_len=test_len,
        optimizer=optimizer,
        logging=True)
print("data_size: train:{}, val:{}, test:{}".format(train_len,val_len,test_len))