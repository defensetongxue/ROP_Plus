import torch
from CNNs.inception_v3 import build_inception3_pretrained as build_model
from utils import generate_data_processer, generate_dataloader, train_process
import torch.optim as optim
from config import paser_args

args = paser_args()
print("Begin pretrain {}".format(args.pretrain))
loss_func = torch.nn.CrossEntropyLoss()
if args.GEN_DATA:
    data_processer = generate_data_processer(PATH=args.PATH,
                                             TEST_DATA=args.TEST_DATA)
    data_processer.generate_test_data()
    data_processer.get_data_condition()

dataloaders, data_auguments, num_class = generate_dataloader(PATH=args.PATH,
                                                             train_proportion=args.train_proportion,
                                                             val_proportion=args.val_proportion,
                                                             batch_size=args.batch_size)
train_loader, val_loader, test_loader = dataloaders
train_len, val_len, test_len = data_auguments

model = build_model(num_classes=num_class,
                    pretrained=args.pretrain)  # TODO model_setting
optimizer = optim.Adam(model.parameters(), lr=args.lr)  # TODO lr decay

train_processer = train_process(model=model, optimizer=optimizer,
                                save_name=args.save_name,
                                epoch=args.epoch,
                                loss_func=loss_func,
                                early_stop=args.early_stop)
print("data_size: train:{}, val:{}, test:{}".format(train_len, val_len, test_len))

train_processer.train(train_loader=train_loader,
                      val_loader=val_loader,
                      train_len=train_len,
                      val_len=val_len,
                      logging=True)

train_process.test(test_loader=test_loader,
                   test_len=test_len)
