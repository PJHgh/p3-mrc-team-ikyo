import json
import pickle
import time
import os

import torch
import torch.nn.functional as F
import random
import numpy as np

from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AdamW
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from datasets import load_from_disk
from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

def get_config():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)


def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()

    return dataset

def get_data(data_path, p_tokenizer, q_tokenizer, training_args, token_data = True):
    if token_data:
        train_dataset = get_pickle(data_path + "/new_train.pkl")
        q_val_dataset = get_pickle(data_path + "/val_q.pkl")
        ground_truth_dataset = get_pickle(data_path + "/new_val_gt.pkl")

    else:
        p_train = load_from_disk(data_path)["train"]["context"]
        q_train = load_from_disk(data_path)["train"]["question"]

        query = load_from_disk(data_path)["validation"]["question"]
        ground_truth = load_from_disk(data_path)["validation"]["context"]
        
        p_train = p_tokenizer(p_train, padding="max_length", truncation=True, return_tensors='pt')
        q_train = q_tokenizer(q_train, padding=True, truncation=True, max_length=100, return_tensors='pt')
        ground_truth_token = p_tokenizer(ground_truth, padding="max_length", truncation=True, return_tensors='pt')
        q_val = q_tokenizer(q_train, padding=True, truncation=True, max_length=100, return_tensors='pt')
        
        train_dataset = TensorDataset(p_train["input_ids"], p_train["attention_mask"], p_train["token_type_ids"],
                                    q_train["input_ids"], q_train["attention_mask"], q_train["token_type_ids"])

        q_val_dataset = TensorDataset(q_val["input_ids"], q_val["attention_mask"], q_val["token_type_ids"])
        ground_truth_dataset = TensorDataset(ground_truth_token["input_ids"], ground_truth_token["attention_mask"], ground_truth_token["token_type_ids"])

    train_iter = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    q_val_iter = DataLoader(q_val_dataset, batch_size=1)
    ground_truth_iter = DataLoader(ground_truth_dataset, batch_size=1)

    return train_dataset, train_iter, q_val_iter, ground_truth_iter


def get_model(model_args, training_args, device):
    p_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    q_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    p_model.to(device)
    q_model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
      {"params" : [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay" : training_args.weight_decay},
      {"params" : [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay" : 0.0},
      {"params" : [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay" : training_args.weight_decay},
      {"params" : [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay" : 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.NLLLoss().to(device)

    return p_model, q_model, optimizer, scaler, scheduler, criterion


def get_passage_emb(p_model, ground_truth_iter):
    p_model.eval()

    pbar = tqdm(enumerate(ground_truth_iter), total=len(ground_truth_iter), position=0, leave=True)

    p_embs = []
    for step, batch in pbar:
        batch = tuple(t.cuda() for t in batch)
        p_inputs = { 
            "input_ids" : batch[0],
            "attention_mask" : batch[1],
            "token_type_ids" : batch[2]
        }
        p_emb = p_model(**p_inputs).pooler_output.to("cpu").numpy()
        p_embs.append(p_emb)
    
    p_embs = torch.Tensor(p_embs).squeeze()
    return p_embs


def test_one_epoch(epoch, p_model, q_model, q_val_iter, ground_truth_iter, device, training_args):

    p_embs = get_passage_emb(p_model, ground_truth_iter)
    q_model.eval()

    pbar = tqdm(enumerate(q_val_iter), total=len(q_val_iter), position=0, leave=True)
    correct = 0
    total = 0
    for step, batch in pbar:
        batch = tuple(t.cuda() for t in batch)

        q_inputs = { 
            "input_ids" : batch[0],
            "attention_mask" : batch[1],
            "token_type_ids" : batch[2]
        }

        q_emb = q_model(**q_inputs).pooler_output.to("cpu")
        
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1 ,descending=True).squeeze()
        #dot_prod_scores.squeeze()[rank[i]]

        if step == rank[0]:
            correct += 1
        total += 1

        description = f"epoch {epoch} acc: {correct/total: .4f}"
        pbar.set_description(description)
    return correct/total
    

def train_one_epoch(epoch, p_model, q_model, optimizer, scaler, scheduler, criterion, train_iter, device, training_args):
    p_model.train()
    q_model.train()

    t = time.time()
    running_loss = 0
    sample_num = 0

    pbar = tqdm(enumerate(train_iter), total=len(train_iter), position=0, leave=True)
    for step, batch in pbar:
        with autocast():
            batch = tuple(t.cuda() for t in batch)
            p_inputs = {
                "input_ids" : batch[0],
                "attention_mask" : batch[1],        
                "token_type_ids" : batch[2]
            }
            q_inputs = { 
                "input_ids" : batch[3],
                "attention_mask" : batch[4],
                "token_type_ids" : batch[5]
            }

            p_outputs = p_model(**p_inputs)
            q_outputs = q_model(**q_inputs)

            sim_scores = torch.matmul(q_outputs.pooler_output, torch.transpose(p_outputs.pooler_output, 0, 1))
            sim_scores = F.log_softmax(sim_scores, dim=1)

            targets = torch.arange(0, len(batch[0])).long().to(device)

            loss = criterion(sim_scores, targets)
            running_loss += loss.item()*training_args.per_device_train_batch_size
            sample_num += training_args.per_device_train_batch_size

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            description = f"epoch {epoch} loss: {running_loss/sample_num: .4f}"
            pbar.set_description(description)

    if scheduler is not None:
        scheduler.step()


def train_dpr(p_model, q_model, optimizer, scaler, scheduler, criterion, train_iter, q_val_iter, ground_truth_iter, device, model_args, data_args, training_args):
    prev_acc = 0
    for epoch in range(int(training_args.num_train_epochs)):
        train_one_epoch(epoch, p_model, q_model, optimizer, scaler, scheduler, criterion, train_iter, device, training_args)
        with torch.no_grad():
            acc = test_one_epoch(epoch, p_model, q_model, q_val_iter, ground_truth_iter, device, training_args)
            if acc > prev_acc:
                torch.save(p_model, training_args.output_dir + "/p_model.pt")
                torch.save(q_model, training_args.output_dir + "/q_model.pt")
                prev_acc = acc


def wiki_emb():
    p_model = torch.load("/opt/ml/lastcode/retrieval_model/p_model.pt")
    with torch.no_grad():
        p_model.eval()
        wiki_dataset = get_pickle("/opt/ml/lastcode/tokenize_data/new_wiki.pkl")
        wiki_iter = DataLoader(wiki_dataset, batch_size=1)
        wiki_embs = []
        pbar = tqdm(enumerate(wiki_iter), total=len(wiki_iter), position=0, leave=True)
        for step, batch in pbar:
            batch = tuple(t.cuda() for t in batch)
            wiki_inputs = { 
                "input_ids" : batch[0],
                "attention_mask" : batch[1],
                "token_type_ids" : batch[2]
            }
            wiki_emb = p_model(**wiki_inputs).pooler_output.to("cpu").numpy()
            wiki_embs.append(wiki_emb)
        
        wiki_embs = torch.Tensor(wiki_embs).squeeze()
        torch.save(wiki_embs, "/opt/ml/lastcode/retrieval_model/wiki_embs.pt")


if __name__ == "__main__":
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_args, data_args, training_args = get_config()
    seed_everything(training_args.seed)

    p_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

    #data_path = "/opt/ml/input/data/data/train_dataset"
    
    data_path = "/opt/ml/lastcode/tokenize_data"

    train_dataset, train_iter, q_val_iter, ground_truth_iter = get_data(data_path, p_tokenizer, q_tokenizer, training_args, token_data=True)
    p_model, q_model, optimizer, scaler, scheduler, criterion = get_model(model_args, training_args, device)

    train_dpr(p_model, q_model, optimizer, scaler, scheduler, criterion, train_iter, q_val_iter, ground_truth_iter, device, model_args, data_args, training_args)
    wiki_emb()