import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from typing import Optional
import numpy as np
import pandas as pd
from nn.model import ANN
from util.utils import CustomDataset
from tqdm.auto import tqdm
import argparse
from eval.validation import *
from tqdm.auto import tqdm
from util.early_stop import EarlyStopper
from metric.loss import RMSLELoss, RMSELoss
from datasets.dataset import get_X, get_y
from metric.graph import get_graph, visualization_metrix
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def train(
  model:nn.Module,
  criterion:callable,
  optimizer:torch.optim.Optimizer,
  data_loader:DataLoader,
  device:str
) -> float:
  '''train one epoch
  
  Args:
      model: model
      criterion: loss
      optimizer: optimizer
      data_loader: data loader
      device: device
  '''
  model.train()
  total_loss = 0.
  for X, y in data_loader:
    X, y = X.to(device), y.to(device)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * len(y)
  return total_loss/len(data_loader.dataset)

def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str,
  metric:Optional[torchmetrics.metric.Metric]=None,
) -> float:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
  '''
  model.eval()
  total_loss,correct = 0.,0.
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      total_loss += criterion(output, y).item() * len(y)
      if metric is not None:
        output = torch.round(output)
        metric.update_state(output, y)

  total_loss = total_loss/len(data_loader.dataset)
  return total_loss 


def main(args):
  device = torch.device(args.device)

  submission_df = pd.read_csv(args.data_submission)
  train_df = pd.read_csv(args.data_train)
  test_df = pd.read_csv(args.data_test)
  X_trn, X_val = get_X(train_df,test_df)
  y_trn = get_y(train_df,test_df)[:,np.newaxis]
  ds = CustomDataset(X_trn.astype(np.float32), y_trn.astype(np.float32))
  ds_val = CustomDataset(X_val.astype(np.float32))
  dl = DataLoader(ds, batch_size=args.batch_size, shuffle=args.shuffle)
  dl_val = DataLoader(ds_val, batch_size=args.batch_size)

  model = ANN(X_trn.shape[-1] ,args.hidden_dim).to(device)
  print(model)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)

  history = {
    'loss':[],
    'val_loss':[],
    'lr':[]
  }
  
  if args.train:
    pbar = range(args.epochs)
    if args.pbar:
      pbar = tqdm(pbar)
    
    print("Learning Start!")
    early_stopper = EarlyStopper(args.patience ,args.min_delta)
    for _ in pbar:
      loss = train(model, nn.MSELoss(), optimizer, dl, device)
      history['lr'].append(optimizer.param_groups[0]['lr'])
      scheduler.step(loss)
      history['loss'].append(loss) 
      pbar.set_postfix(trn_loss=loss)
      if early_stopper.early_stop(model, loss, args.output+args.name+'_earlystop.pth'):
        print('Early Stopper run!')            
        break
    get_graph(history, args.name)  
    print("Done!")
    torch.save(model.state_dict(), args.output+args.name+'.pth')
    
    model = ANN(X_trn.shape[-1] ,args.hidden_dim).to(device)
    if torch.load(args.output+args.name+'_earlystop.pth'):
      model.load_state_dict(torch.load(args.output+args.name+'_earlystop.pth'))
    else:
      model.load_state_dict(torch.load(args.output+args.name+'.pth'))
    model.eval()
    
    pred = []
    with torch.inference_mode():
      for x in dl_val:
        x = x[0].to(device)
        out = model(x)
        pred.append(out.detach().cpu().numpy())
    
    #visualization_metrix(model,torch.tensor(X_trn).detach().cpu().numpy(),y_trn,args.name)
    submission_df['SalePrice'] = np.concatenate(pred).squeeze()+12088.66311880681
    submission_df.to_csv(args.submission+args.name+'.csv',index=False)
  
  print('------------------------------------------------------------------')
  if args.validation:
    model = ANN(X_trn.shape[-1] ,args.hidden_dim).to(device)
    scores = Validation(X_trn, y_trn, args.patience, args.min_delta)
    scores = pd.DataFrame(scores.kfold(model, n_splits=5, epochs=args.epochs, lr=args.lr, batch=args.batch_size, shuffle=True, random_state=2023))
    print(pd.concat([scores, scores.apply(['mean', 'std'])]))
    
  return


def get_args_parser(add_help=True):

  parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

  parser.add_argument("--data-submission", default="../data/sample_submission.csv", type=str, help="submission dataset path")
  parser.add_argument("--data-train", default="../data/train.csv", type=str, help="train dataset path")
  parser.add_argument("--data-test", default="../data/test.csv", type=str, help="test dataset path")
  parser.add_argument("--hidden-dim", default=64, type=int, help="dimension of hidden layer")
  parser.add_argument("--device", default="cpu", type=str, help="device (Use cpu/cuda/mps)")
  parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
  parser.add_argument("--shuffle", default=True, type=bool, help="shuffle")
  parser.add_argument("--epochs", default=2000, type=int, metavar="N", help="number of total epochs to run")
  parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
  parser.add_argument("--pbar", default=True, type=bool, help="progress bar")
  parser.add_argument("-o", "--output", default="./submit/model_", type=str, help="path to save output model")
  parser.add_argument("-sub", "--submission", default="./submit/submission_", type=str, help="path to save submission")
  parser.add_argument("-train", "--train", default=False, type=bool, help="full data set train")
  parser.add_argument("-val", "--validation", default=False, type=bool, help="kfold cross validation train")
  parser.add_argument("-pat", "--patience", default=200, type=int, help="Early stop patience count")
  parser.add_argument("-delta", "--min-delta", default=0, type=int, help="Early stop delta value")
  parser.add_argument("-name", "--name", default="", type=str, help="model name for Outputs")
  
  
  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  main(args)