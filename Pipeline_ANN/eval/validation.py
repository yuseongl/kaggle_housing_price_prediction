import torch
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
import tensorflow as tf
from tqdm.auto import tqdm
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from train import train, evaluate 
from copy import deepcopy
from util.early_stop import EarlyStopper
from metric.rmsle import RMSLELoss, RMSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings(action='ignore')

class Validation:
  def __init__(self, X_trn, y_trn, patience, delta):
    self.patience = patience
    self.delta = delta
    self.X, self.y = 0,0 
    self.X_trn = X_trn
    self.y_trn = y_trn
    self.pred = 0
    self.scores={
    'MSE':[],
    'RMSE':[],
    'RMSLE':[],
    'MAE':[],
    'R2SCORE':[]
    }
    return
  
  def kfold(self, model, n_splits, shuffle=True, lr=0.001, epochs=100, batch=64, random_state=2023, device='cpu'):
    X_val, y_val = 0,0
    n_splits = n_splits

    skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for i, (trn_idx, val_idx) in enumerate(skf.split(self.X_trn, self.y_trn)):
      self.X, self.y = torch.tensor(self.X_trn[trn_idx]), torch.tensor(self.y_trn[trn_idx])
      X_val, y_val = torch.tensor(self.X_trn[val_idx]), torch.tensor(self.y_trn[val_idx])

      ds = TensorDataset(self.X, self.y)
      ds_val = TensorDataset(X_val, y_val)
      # ds = CustomDataset(X, y)
      # ds_val = CustomDataset(X_val, y_val)
      dl = DataLoader(ds, batch, shuffle=True)
      dl_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)
      
      net = deepcopy(model)
      net.to(device)
      
      optimizer = torch.optim.AdamW(net.parameters(), lr)
      #scheduler = ReduceLROnPlateau(optimizer,'min',factor=0.8,patience=3,min_lr=0.000001)
      scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=0.00001)

      
      pbar = range(epochs)
      pbar = tqdm(pbar)
      early_stopper = EarlyStopper(self.patience, self.delta)
      for j in pbar:
        accuracy = tf.keras.metrics.Accuracy()
        loss = train(net, nn.MSELoss(), optimizer, dl, device)
        loss_val = evaluate(net, nn.MSELoss(), dl_val, device, accuracy)
        scheduler.step(loss)
        acc_val = accuracy.result().numpy()
        pbar.set_postfix(trn_loss=loss, val_loss=loss_val, val_acc=acc_val)
        if early_stopper.early_stop(net,validation_loss=loss_val, mode=False):             
          break

      self.pred = net(self.X)
      #self.pred = torch.round(self.pred)
      self.metric()
      del net
      
    return self.scores
  
  def metric(self):
    y_true = self.y.detach().numpy()
    y_pred = self.pred.detach().numpy()

    MSE = mean_squared_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred,squared=False)
    RMSLE = mean_squared_log_error(y_true, y_pred,squared=False)
    MAE = mean_absolute_error(y_true, y_pred)
    R2SCORE = r2_score(y_true, y_pred)

    self.scores['MSE'].append(MSE)
    self.scores['RMSE'].append(RMSE)
    self.scores['RMSLE'].append(RMSLE)
    self.scores['MAE'].append(MAE)
    self.scores['R2SCORE'].append(R2SCORE)
    
    return 
  
  def __call__(self):
    
    return