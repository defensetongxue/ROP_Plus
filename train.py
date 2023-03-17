import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


class train_process():
    def __init__(self, model, optimizer, epoch,
                 save_name='best.pth',
                 loss_func=nn.CrossEntropyLoss, early_stop=1000):
        super(train_process, self).__init__()
        self.epoch = epoch
        self.save_path = 'save_models/{}'.format(save_name)
        self.loss_func = loss_func
        self.early_stop = early_stop
        self.model = model.cuda()
        self.optimizer = optimizer

    def train(self, train_loader, val_loader, test_loader, train_len, val_len, test_len,
              logging=False):
        print("begin_trainning process")
        patient = 0
        best_loss = 1e10
        for epoch in range(self.epoch):
            self.model.train()
            train_loss = 0
            pred_train = []
            label_train = []
            score_train = []
            for batch_x, batch_y in train_loader:
                logits, loss, pred = self.train_epoch(batch_x, batch_y)
                score_train.append(logits.cpu())
                pred_train.append(pred.cpu())
                label_train.append(batch_y.cpu())
                train_loss += loss.data.item()
            # raise
            score_train = torch.cat(score_train, dim=0)
            score_train = F.softmax(score_train, dim=-1)
            pred_train = torch.cat(pred_train, dim=0)
            label_train = torch.cat(label_train, dim=0)
            mean_loss_train = train_loss/train_len
            mean_accu_train = (pred_train == label_train).sum()/train_len
            auc_train = roc_auc_score(
                label_train, score_train.detach(), multi_class='ovo')
            # raise
            # evaluation
            self.model.eval()
            eval_loss = 0
            pred_val = []
            label_val = []
            score_val = []
            for batch_x, batch_y in val_loader:
                logits, loss, pred = self.val_epoch(batch_x, batch_y)
                eval_loss += loss.data.item()
                score_val.append(logits.cpu())
                pred_val.append(pred.cpu())
                label_val.append(batch_y.cpu())
            score_val = torch.cat(score_val, dim=0)
            score_val = F.softmax(score_val, dim=-1)
            pred_val = torch.cat(pred_val, dim=0)
            label_val = torch.cat(label_val, dim=0)
            mean_loss_val = eval_loss/val_len
            mean_accu_val = (pred_val == label_val).sum()/val_len
            auc_val = roc_auc_score(
                label_val, score_val.detach(), multi_class='ovo')
            if logging:
                print("Epoch {} Train: loss:{:.6f}, Acc:{:.6f}, Auc:{:.6f}|".format(
                    epoch, mean_loss_train, mean_accu_train, auc_train) +
                    " Val: loss:{:.6f}, Acc:{:.6f}, Auc:{:.6f}".format(
                    mean_loss_val, mean_accu_val, auc_val))
            patient += 1
            if mean_loss_val < best_loss:
                best_loss = mean_loss_val
                patient = 0
                torch.save(self.model.state_dict(), self.save_path)
            if patient >= self.early_stop:
                break

        # test
        self.model.load_state_dict(
            torch.load(self.save_path))
        self.model.eval()
        pred_test = []
        label_test = []
        score_test = []
        for batch_x, batch_y in test_loader:
            logits, loss, pred = self.test_epoch(batch_x, batch_y)
            score_test.append(logits.cpu())
            pred_test.append(pred.cpu())
            label_test.append(batch_y.cpu())
        
        score_test = torch.cat(score_test, dim=0)
        score_test = F.softmax(score_test, dim=-1)
        pred_test = torch.cat(pred_test, dim=0)
        label_test = torch.cat(label_test, dim=0)
        mean_accu_test = (pred_test == label_test).sum()/test_len
        auc_test = roc_auc_score(
            label_test, score_test.detach(), multi_class='ovo')
        print("Test: acc:{:.6f} auc:{:.6f}".format(mean_accu_test, auc_test))

    def train_epoch(self, batch_x, batch_y):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        logits, aux = self.model(batch_x)
        loss = self.loss_func(logits, batch_y) + \
            self.loss_func(aux, batch_y)
        pred = torch.max(logits, 1)[1]
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return logits, loss, pred

    def val_epoch(self, batch_x, batch_y):
        with torch.no_grad():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            logits = self.model(batch_x)
            loss = self.loss_func(logits.cpu(), batch_y.cpu())
            pred = torch.max(logits, 1)[1]
            return logits, loss, pred

    def test_epoch(self,batch_x, batch_y):
        with torch.no_grad():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            logits = self.model(batch_x)
            loss = self.loss_func(logits.cpu(), batch_y.cpu())
            pred = torch.max(logits, 1)[1]
            return logits, loss, pred
    def test_only(self,test_loader,test_len):
        self.model.load_state_dict(
            torch.load(self.save_path))
        self.model.eval()
        pred_test = []
        label_test = []
        score_test = []
        for batch_x, batch_y in test_loader:
            logits, loss, pred = self.test_epoch(batch_x, batch_y)
            score_test.append(logits.cpu())
            pred_test.append(pred.cpu())
            label_test.append(batch_y.cpu())

        score_test = torch.cat(score_test, dim=0)
        score_test = F.softmax(score_test, dim=-1)
        pred_test = torch.cat(pred_test, dim=0)
        label_test = torch.cat(label_test, dim=0)
        mean_accu_test = (pred_test == label_test).sum()/test_len
        auc_test = roc_auc_score(
            label_test, score_test.detach(), multi_class='ovo')
        print("Test: acc:{:.6f} auc:{:.6f}".format(mean_accu_test, auc_test))