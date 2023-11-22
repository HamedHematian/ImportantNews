# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
import pandas as pd
from copy import deepcopy
import sys
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import math
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import seaborn as sn
import matplotlib.pyplot as plt
import random
import gdown
import os
import ast
from sklearn.model_selection import train_test_split
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch.backends.cudnn
import torch.cuda

gdown.download(id='1ViHBOU4WyYzIGNpfjewxwesa7Dn-Edp_')
gdown.download(id='156KELVr_LtK8UiDDCf0U3H5uf-NPFBMa')
gdown.download(id='1IYF7LeOBnl6H7EBf7mvabXd8G4sGeyKp')

SEED = 12345
def set_determenistic_mode(SEED, disable_cudnn=False):
  torch.manual_seed(SEED)                       # Seed the RNG for all devices (both CPU and CUDA).
  random.seed(SEED)                             # Set python seed for custom operators.
  rs = RandomState(MT19937(SeedSequence(SEED))) # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
  np.random.seed(SEED)
  torch.cuda.manual_seed_all(SEED)              # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

  if not disable_cudnn:
    torch.backends.cudnn.benchmark = False    # Causes cuDNN to deterministically select an algorithm,
                                              # possibly at the cost of reduced performance
                                              # (the algorithm itself may be nondeterministic).
    torch.backends.cudnn.deterministic = True # Causes cuDNN to use a deterministic convolution algorithm,
                                              # but may slow down performance.
                                              # It will not guarantee that your training process is deterministic
                                              # if you are using other libraries that may use nondeterministic algorithms
  else:
    torch.backends.cudnn.enabled = False # Controls whether cuDNN is enabled or not.
                                         # If you want to enable cuDNN, set it to True.
set_determenistic_mode(SEED)
def seed_worker(worker_id):
    worker_seed = SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(SEED)

train_data = pd.read_csv('train_data.csv')
eval_data = pd.read_csv('eval_data.csv')
test_data = pd.read_csv('test_data.csv')
class_weight = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=train_data['last_sentiment'].values).tolist()
print(len(train_data))




class NewsDataset(Dataset):

  def __init__(self, data, tokenizer, use_text=True, use_title=True, use_keywords=True, max_length=512):
    self.data = data
    self.use_text = use_text
    self.use_title = use_title
    self.use_keywords = use_keywords
    self.max_length = tokenizer.max_len_single_sentence
    self.cls_token = tokenizer.cls_token
    self.sep_token = tokenizer.sep_token
    self.max_length = max_length

  def __len__(self):
    return len(self.data)

  def tokenize(self, input_1, input_2):
    if input_2 is None:
      return tokenizer(input_1, padding='max_length', max_length=self.max_length, truncation='only_second', return_tensors='pt')
    else:
      return tokenizer(input_1, input_2, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')

  def __getitem__(self, idx):
    data = self.data.loc[idx]
    if str(data['tags']) == 'nan':
      keywords = ''
    else:
      keywords = ast.literal_eval(data['tags'])
      keywords = ','.join(keywords)
    if self.use_keywords and self.use_title and self.use_text:
      input = self.tokenize(data['title'] + ',' + keywords, data['text'])
    elif self.use_keywords and self.use_title:
      input = self.tokenize(keywords, data['title'])
    elif self.use_title and self.use_text:
      input = self.tokenize(data['title'], data['text'])
    elif self.use_keywords and self.use_text:
      input = self.tokenize(keywords, data['text'])
    elif self.use_keywords:
      input = self.tokenize(keywords, None)
    elif self.use_title:
      input = self.tokenize(data['title'], None)
    elif self.use_text:
      input = self.tokenize(data['text'])
    input['label'] = torch.tensor(data['last_sentiment'])

    return input

  @staticmethod
  def collate(batch):
    pass

class TransformerModel(nn.Module):

  def __init__(self, tarnsformer, hidden_size=400):
    super(TransformerModel, self).__init__()
    self.transformer = transformer
    self.linear_head = nn.Linear(transformer.config.hidden_size, 2)

  def forward(self, x):
    x = self.transformer(**x).pooler_output
    x = self.linear_head(x)
    return x
    
def train(epoch, model, dataloader, optimizer, scheduler, device):
    model.train()
    loss_collection = []
    step = 0
    for data in dataloader:
      step += 1
      label = data.pop('label')
      for key in data.keys():
        data[key] = data[key].squeeze(1).to(device)
      loss_fn = nn.CrossEntropyLoss(torch.tensor(class_weight).to(device).float())
      output = model(data)
      loss = loss_fn(output, label.to(device))
      loss_collection.append(loss.item())
      # optimization
      loss.backward()
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      if len(loss_collection) and len(loss_collection) % 100 == 0:
        loss_ = sum(loss_collection) / len(loss_collection)
        loss_collection = []
        epochs = params['epochs']
        print(f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step}/{len(dataloader)}] | Loss {round(loss_, 2)}')

def eval(epoch, model, dataloader, device):
  model.eval()
  with torch.no_grad():
    all_pred = []
    all_label = []
    global g
    for data in dataloader:
      g = deepcopy(data)
      label = data.pop('label')
      for key in data.keys():
        data[key] = data[key].squeeze(1).to(device)

      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)
      all_label.extend(list(label.detach().cpu().numpy()))
      all_pred.extend(list(pred.view(-1).detach().cpu().numpy()))

    output = dict()
    output['f1_macro'] = f1_score(all_label, all_pred, average='macro')
    output['f1_micro'] = f1_score(all_label, all_pred, average='micro')
    output['accuracy'] = accuracy_score(all_label, all_pred)
    output['recall_macro'] = recall_score(all_label, all_pred, average='macro')
    output['recall_micro'] = recall_score(all_label, all_pred, average='micro')
    output['precision_macro'] = precision_score(all_label, all_pred, average='macro')
    output['precision_micro'] = precision_score(all_label, all_pred, average='micro')
    output['all_pred'] = all_pred
    output['all_label'] = all_label
    return output

weight_decay = 1e-2
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
transformer_path = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(transformer_path)
transformer = AutoModel.from_pretrained(transformer_path)
batch_size = 10

# different settings
use_keywords = True
use_title = True
use_text = True
max_length = 512
best_f1 = 0.
all_f1 = list()
best_model = None
best_preds = None
best_output = None

def save_model(model, filename):
  torch.save(
      {
       'model_state_dict': model.state_dict(),
       }, filename)

params = {
    'epochs': 13,
    'hidden_size': 400,
    'warmup_ratio': .0

}

tuning_params = [
    {'lr': 1e-5},
    {'lr': 8e-6},
    {'lr': 2e-5}]
eval_turn = int(sys.argv[1])
eval_turn_str = str(eval_turn)
print(tuning_params[eval_turn])



# define model
model = TransformerModel(deepcopy(transformer), params['hidden_size']).to(device)
# train dataloader
train_dataset = NewsDataset(train_data, tokenizer, use_keywords, use_title, use_text, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# eval dataloader
eval_dataset = NewsDataset(eval_data, tokenizer, use_keywords, use_title, use_text, max_length)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
# eval dataloader
test_dataset = NewsDataset(test_data, tokenizer, use_keywords, use_title, use_text, max_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# optimizations
optimization_steps = params['epochs'] * len(train_dataloader)
warmup_steps = int(params['warmup_ratio'] * optimization_steps)
optimizer = AdamW(model.parameters(), lr=tuning_params[eval_turn]['lr'], weight_decay=1e-3)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=optimization_steps)

for epoch in range(params['epochs']):
  train(epoch, model, train_dataloader, optimizer, scheduler, device)
  output = eval(epoch, model, eval_dataloader, device)
  f1_macro = output['f1_macro']
  all_f1.append(f1_macro)
  if f1_macro > best_f1:
    best_f1 = f1_macro
    best_output = output
    save_model(model, 'best_model.pt')
  print('score', output['f1_macro'])
  print('best score', best_f1)


dir_ = 'drive/MyDrive/xlmr_0/'
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])
output = eval(epoch, model, test_dataloader, device)
for key, value in output.items():
  print(key, value)


