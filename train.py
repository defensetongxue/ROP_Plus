import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
from sklearn.metrics import roc_auc_score


class train_process():
    def __init__(self, epoch, lr=1e-3, loss_func=nn.CrossEntropyLoss):
        self.epoch = epoch
        self.lr = lr
        self.loss_func = loss_func

    def train(self, model, train_loader, test_loader, train_len, test_len, 
            optimizer=None, logging=False,save_model=False):
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        print("begin_trainning process")
        for epoch in range(self.epoch):
            model.train()
            train_loss = 0
            train_accu = 0
            auc_train = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = (batch_x).cuda(), (batch_y).cuda()
                out = model(batch_x)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                loss = self.loss_func(out, batch_y)
                train_loss += loss.data.item()
                pred = torch.max(out, 1)[1]

                train_correct = (pred == batch_y).sum()
                train_accu += train_correct.data.item()
                auc_train += roc_auc_score(pred.cpu(), batch_y.cpu()).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss = train_loss/train_len
            mean_accu = train_accu/train_len
            auc_train = auc_train/train_len
            if logging:
                print('epoch :{} Training Loss : {:.6f}, Accu: {:.6f}, Auc:{:.6f}'.format(
                    epoch, mean_loss, mean_accu, auc_train))

            # evaluation
            model.eval()
            with torch.no_grad():
                eval_loss = 0
                eval_accu = 0
                auc_score = 0
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = Variable(
                        batch_x).cuda(), Variable(batch_y).cuda()
                    out = model(batch_x)
                    loss = self.loss_func(out.cpu(), batch_y.cpu())
                    eval_loss += loss.data.item()
                    pred = torch.max(out, 1)[1]
                    num_correct = (pred == batch_y).sum()
                    # step_auc_score=roc_auc_score(pred,batch_y).sum()
                    eval_accu += num_correct.data.item()
                    # auc_score += step_auc_score
                mean_loss = eval_loss/test_len
                mean_accu = eval_accu/test_len
                auc_score =auc_score/test_len
                if logging:
                    print('Testing Loss:{:.6f},Accu:{:.6f},Auc:{:.6f}'.format(
                        mean_loss,mean_accu,auc_score))

            if save_model:
                name = 'inception3'+'_'+str(epoch)+'.pkl'
                torch.save(model, './models/'+name)
