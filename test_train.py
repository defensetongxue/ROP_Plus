

from test_model import Inception3
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from  torchvision import transforms,utils
from torch.utils.data import DataLoader
import  torch
import pdb

import dataloader 
pos_cnt,neg_cnt=dataloader.generate_test_data(TEST_DATA=6400)
print(pos_cnt,neg_cnt)
train_loader,test_loader,train_data,test_data=dataloader.generate_dataloader(batch_size=64)
model = Inception3()#使用gpu，将模型加载到显存
model = model.cuda()

loss_func = nn.CrossEntropyLoss()
print("begin_trainning process")
for epoch in range(10):

	#training------------------------
	train_loss = 0
	train_accu = 0
	step = 0
	for batch_x,batch_y in train_loader:
		batch_x,batch_y = Variable(batch_x).cuda(),Variable(batch_y).cuda()#数据加载到显存
		out = model(batch_x)
		out1 = out[0]

		optimizer = optim.Adam(model.parameters(),lr=0.001)
		loss = loss_func(out1,batch_y)
		print(loss.data)
		train_loss += loss.data.item()
		pred = torch.max(out1,1)[1]
		train_correct = (pred==batch_y).sum()
		train_accu += train_correct.data.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print("EPOCH:%d,STEP:%d"%(epoch,step))
		step = step + 1
	mean_loss = train_loss/(len(train_data))
	mean_accu = train_accu/(len(train_data))
	#print(mean_loss,mean_accu)
	print('Training Loss : %.6f,Accu: %.6f'%(mean_loss,mean_accu))
											
	#evaluation------------------------
	model.eval()
	eval_loss = 0
	eval_accu = 0
	for batch_x,batch_y in test_loader:
		batch_x,batch_y = Variable(batch_x,volatile=True).cuda(),Variable(batch_y,volatile=True).cuda()
		out = model(batch_x)
		loss = loss_func(out,batch_y)
		eval_loss += loss.data.item()
		pred = torch.max(out,1)[1]
		num_correct = (pred == batch_y).sum()
		eval_accu += num_correct.data.item()
	mean_loss = eval_loss/(len(test_data))
	mean_accu = eval_accu/(len(test_data))
	print('Testing Loss:%.6f,Accu:%.6f'%(mean_loss,mean_accu))

	name = 'inception3'+'_'+str(epoch)+'.pkl'
	torch.save(model,'./models/'+name)