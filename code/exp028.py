# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import configparser
import json
import time
import math
import string
import pickle
import random
import joblib
import logging
import itertools
import datetime
import warnings
import shutil
import requests
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error # 評価関数として設定
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

#os.system('pip install iterative-stratification==0.1.7')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import slackweb

import torch
import torch.nn as nn
import torch.utils.checkpoint # version 1.12.0では必要
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
print(f"torch.__version__: {torch.__version__}")
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

# ====================================================
# ENVIRONMENT
# ====================================================
config_ini = configparser.ConfigParser()
config_ini.read("../config/config.ini", encoding="utf-8")
WANDB_API_KEY = config_ini["WANDB_API"]["API_KEY"]
NOTION_API_KEY = config_ini["NOTION_API"]["API_KEY"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================================================
# CFG
# ====================================================
class CFG:
    name = "exp028"
    explain = "deberta-v3-base re-initial layer 2"
    wandb = True
    send_slack = False
    send_notion = True
    competition = "FB3"
    _wandb_kernel = "kfksy"
    debug = False
    apex = True
    print_freq = 100
    num_workers = 4
    gradient_checkpointing=True
    model = "microsoft/deberta-v3-base"
    scheduler = "CosineAnnealingLR"
    batch_scheduler = True
    num_cycle = 0.5
    num_warmup_steps = 100
    epochs = 5
    encoder_lr = 3e-5
    decoder_lr = 3e-5
    min_lr = 1e-7
    eps = 1e-7
    min_lr = 1e-7
    betas = (0.9, 0.999)
    reinit_layers = 2
    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    T_max=50 # CosineAnnealingLR
    #T_0=50 # CosineAnnealingWarmRestarts
    batch_size=4 # https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/308298
    fc_dropout=0.2
    max_len=2048
    weight_decay=0.01
    #target_size=3
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True
    wandb_key = WANDB_API_KEY

    #slack_path = "https://hooks.slack.com/services/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" # 共有するときは削除
    notion_api = "secret_mDRpQpcMtlPJiaLh4jG7MHDzJK6iAAjQixkEwVwK3yC"

    # Kaggle upload
    upload_from_colab = True
    
if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]

# ====================================================
# Logger
# ====================================================
class Logger:
    def __init__(self, path):
        self.general_logger = logging.getLogger(path)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, "Experiment.log"))
        if len(self.general_logger.handlers) ==0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        # display time
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# ====================================================
# MAKE DIR
# ====================================================
# set dir
INPUT = "../input"
OUTPUT= "../output"
EXP = (CFG.name if CFG.name is not None 
       else get("http://172.28.0.2:9000/api/sessions").json()[0]["name"][:-6])

OUTPUT_EXP = os.path.join(OUTPUT, EXP) 
EXP_MODEL = os.path.join(OUTPUT_EXP, "model")
# EXP_FIG = os.path.join(OUTPUT_EXP, "fig")
EXP_PREDS = os.path.join(OUTPUT_EXP, "preds")
EXP_TOKENIZER = os.path.join(OUTPUT_EXP, "tokenizer") # change

# make dirs
for d in [INPUT, EXP_MODEL, EXP_PREDS]:
    os.makedirs(d, exist_ok=True)

logger = Logger(OUTPUT_EXP)

# ====================================================
# wandb
# ====================================================
if CFG.wandb:
    import wandb

    try:
        wandb.login(key=CFG.wandb_key)
        anory = None
    except:
        anory = "must"
        print("please check wandb key")

    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    run = wandb.init(project="FB3",
                   name=CFG.model,
                   config=class2dict(CFG),
                   group=CFG.model,
                   job_type="train",
                   anonymous=anory)

# ====================================================
# Utils
# ====================================================
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label
    

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = torch.stack(list(all_hidden_states), dim=0)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

# max pooling
class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings
    

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            logger.info(self.config)
        else:
            self.config = torch.load(config_path)
            
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
            
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        if cfg.reinit_layers > 0:
            print(f'Reinitializing Last {cfg.reinit_layers} Layers ...')
            #encoder_temp = getattr(model, _model_type)
            for layer in self.model.encoder.layer[-cfg.reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
            print('Done.!')

        self.max_pool = MaxPooling() # change
        self.mean_pool = MeanPooling() # change
        self.weighted_pooler = WeightedLayerPooling(num_hidden_layers=self.config.num_hidden_layers, layer_start=7)
        self.layernorm = nn.LayerNorm(self.config.hidden_size)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc = nn.Linear(self.config.hidden_size*2, 6) 
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        all_hidden_states = outputs[1]
        
        last_hidden_states = outputs[0]
        
        mean_pool = self.mean_pool(last_hidden_states, inputs['attention_mask'])
        
        _weighted_pool = self.weighted_pooler(all_hidden_states)
        weighted_pool = _weighted_pool[:, 0]
        
        feature = torch.cat([weighted_pool, mean_pool], dim=1)

        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        
        output = self.fc(feature)
        
        return output

# ====================================================
# Loss
# ====================================================
class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        if CFG.wandb:
            wandb.log({f"[fold{fold}] loss": losses.val,
                       f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):
    
    logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, os.path.join(OUTPUT_EXP, 'config.pth'))
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
    # ====================================================
    # scheduler
    # ====================================================

    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
              optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
          )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
              optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
          )
        elif cfg.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif cfg.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif cfg.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    
    # ====================================================
    # loop
    # ====================================================
    criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")
    
    best_score = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        
        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        logger.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        logger.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        if CFG.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch+1, 
                       f"[fold{fold}] avg_train_loss": avg_loss, 
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score})
        
        if best_score > score:
            best_score = score
            logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                  'predictions': predictions},
                 os.path.join(EXP_MODEL,f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth"))

    predictions = torch.load(os.path.join(EXP_MODEL,f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth"), 
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

def get_result(oof_df):
        labels = oof_df[CFG.target_cols].values
        preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
        score, scores = get_score(labels, preds)
        logger.info(f'Score: {score:<.4f}  Scores: {scores}')
        return score

# ====================================================
# MAIN
# ====================================================
def main():
    seed_everything(seed=42)

	# ====================================================
	# Data Loading
	# ====================================================
    train = pd.read_csv(os.path.join(INPUT, "train.csv"))
    test = pd.read_csv(os.path.join(INPUT, "test.csv"))
    submission = pd.read_csv(os.path.join(INPUT, "sample_submission.csv"))

	# ====================================================
	# CV split
	# ====================================================
    Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_cols])):
        train.loc[val_index, 'fold'] = int(n)

    train['fold'] = train['fold'].astype(int)
    
    if CFG.debug:
        train = train.sample(n=1000, random_state=0).reset_index(drop=True)

	# ====================================================
	# tokenizer
	# ====================================================
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(EXP_TOKENIZER)
    CFG.tokenizer = tokenizer

	# ====================================================
	# Define max_len
	# ====================================================
    lengths = []
    tk0 = tqdm(train['full_text'].fillna("").values, total=len(train))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    #CFG.max_len = max(lengths) + 3 # cls & sep & sep
    logger.info(f"max_len: {CFG.max_len}")
    
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        logger.info(f"========== CV ==========")
        score = get_result(oof_df)
        oof_df.to_pickle(os.path.join(EXP_PREDS, 'oof_df.pkl')) 
        
    if CFG.wandb:
        wandb.finish()

    # upload output folder to kaggle dataset
    if CFG.upload_from_colab:
        f = open("../config/kaggle.json", 'r')
        json_data = json.load(f) 
        os.environ["KAGGLE_USERNAME"] = json_data["username"]
        os.environ["KAGGLE_KEY"] = json_data["key"]
        
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        def dataset_create_new(dataset_name, upload_dir):
            dataset_metadata = {}
            dataset_metadata['id'] = f'{os.environ["KAGGLE_USERNAME"]}/{dataset_name}'
            dataset_metadata['licenses'] = [{'name': 'CC0-1.0'}]
            dataset_metadata['title'] = dataset_name
            with open(os.path.join(upload_dir, 'dataset-metadata.json'), 'w') as f:
                json.dump(dataset_metadata, f, indent=4)
            api = KaggleApi()
            api.authenticate()
            api.dataset_create_new(folder=upload_dir, convert_to_csv=False, dir_mode='tar')
            
        dataset_create_new(dataset_name=CFG.competition + "-" + CFG.name, upload_dir=OUTPUT_EXP)

    if CFG.send_notion:
        url = f"https://api.notion.com/v1/pages"
        database_id = "93d0f104b15e4dab83b24b4aa26fe563"
        
        headers = {"Authorization": f"Bearer {CFG.notion_api}",
                   "Content-Type": "application/json",
                   "Notion-Version": "2021-05-13"
                  }
        body = {
            "parent": {
                "database_id": database_id
            },
            "properties": {
                "Name": {"title": [{"text": {"content": str(CFG.name)}}]},
                "model": {"rich_text":[{"text": {"content": str(CFG.model)}}]},
                "max_len": {"rich_text":[{"text": {"content": str(CFG.max_len)}}]},
                "fold": {"rich_text":[{"text": {"content": str(CFG.n_fold)}}]},
                "eps": {"rich_text":[{"text": {"content": str(CFG.eps)}}]},
                "scheduler": {"rich_text":[{"text": {"content": str(CFG.scheduler)}}]},
                "batch_size": {"rich_text":[{"text": {"content": str(CFG.batch_size)}}]},
                "coments": {"rich_text":[{"text": {"content": str(CFG.explain)}}]},
                "score": {"rich_text":[{"text": {"content": str(round(score,4))}}]},
            }
        }
        response = requests.request('POST', url=url, headers=headers, data=json.dumps(body))

if __name__ == "__main__":
    main()
