import torch
import numpy as np
import time 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import wandb
import os 
class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion,
        threshold = 0.5
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.threshold = threshold

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None
        
    def fit(self, epochs, train_loader, valid_loader, save_path, patience):        
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)
            
            train_loss, train_time = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time, prec, recall, f = self.valid_epoch(valid_loader)
            
            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s            ",
                n_epoch, train_loss, train_time
            )
            
            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s, precision: {:.2f}, recall: {:.2f}, f: {:.2f}",
                n_epoch, valid_loss, valid_auc, valid_time, prec, recall, f
            )

            # if True:
            if self.best_valid_score < valid_loss: 
                self.save_model(n_epoch, save_path, valid_loss, valid_auc)
                self.info_message(
                     "auc improved from {:.4f} to {:.4f}. Saved model to '{}'", 
                    self.best_valid_score, valid_loss, self.lastmodel
                )
                self.best_valid_score = valid_loss
                self.n_patience = 0
                # torch.save({'epoch': n_epoch,
                #   'model_state_dict': self.model.state_dict(),
                #   'optimizer_state_dict': self.optimizer.state_dict(),
                #   'loss': train_loss,
                #   'F1 score': f}, 
                #    save_path + f"/model_checkpoint_{n_epoch}.pth"
                # )
                torch.save(self.model, save_path+"checkpoint_"+str(n_epoch)+".pt")
                # wandb.save('checkpoint.pth')
                torch.save(self.model, os.path.join(wandb.run.dir, "model.pt"))
            else:
                self.n_patience += 1
            
            if self.n_patience >= patience:
                self.info_message("\nValid auc didn't improve last {} epochs.", patience)
                break
            # TODO: Remove this save later
            # torch.save(self.model, save_path+"/checkpoint_"+str(n_epoch)+".pt")
            
            
    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        sum_loss = 0

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)
            
            loss = self.criterion(outputs, targets)
            loss.backward()

            sum_loss += loss.detach().item()

            self.optimizer.step()
            message = 'Train Step {}/{}, train_loss: {:.4f}'
            self.info_message(message, step, len(train_loader), sum_loss/step, end="\r")
            
        wandb.log({"train_loss": sum_loss/len(train_loader)})
        return sum_loss/len(train_loader), int(time.time() - t)
    
    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()
                y_all.extend(batch["y"].tolist())
                outputs_all.extend(torch.sigmoid(outputs).tolist())

            message = 'Valid Step {}/{}, valid_loss: {:.4f}'

            self.info_message(message, step, len(valid_loader), sum_loss/step, end="\n")
            
        # y_all = [1 if x > 0.5 else 0 for x in y_all]
        outputs_all =  [1 if x > self.threshold else 0 for x in outputs_all]
        print(outputs_all)
        auc = roc_auc_score(y_all, outputs_all)
        #precision
        precision = precision_score(y_all,outputs_all)
        #recall
        recall = recall_score(y_all,outputs_all)
        # F1
        f1 = f1_score(y_all,outputs_all)
        wandb.log({"AUC": auc, "Precision": precision, "Recall": recall, "F1-Score": f1})
        
        return sum_loss/len(valid_loader), auc, int(time.time() - t), precision, recall, f1
    
    def save_model(self, n_epoch, save_path, loss, auc):
        self.lastmodel = f"{save_path}-best.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.lastmodel,
        )
    
    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)
        