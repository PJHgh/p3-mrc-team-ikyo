import os, copy, time, json, random, pickle, argparse
import numpy as np

from konlpy.tag import Mecab
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from datasets import load_dataset, load_from_disk
from transformers import (BertConfig,
                          BertModel,
                          BertPreTrainedModel,
                          AdamW,
                          TrainingArguments,
                          get_linear_schedule_with_warmup,
                          TrainingArguments,
                          set_seed,
                          AutoTokenizer)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

seed_everything(seed=2021)


org_dataset = load_from_disk("/opt/ml/input/data/data/train_dataset")
print("*" * 40, "query dataset", "*" * 40)
print(org_dataset)


training_dataset = org_dataset['train']
validation_dataset = org_dataset['validation']

mecab = Mecab()

def morphs_split(text):
    text = mecab.morphs(text)
    return ' '.join(text)

def context_split(text):
    text = ' '.join(text.strip().splir('\\n')).strip()
    sent_list = text.strip().split('. ')
    text = ''
    for sent in sent_list:
        sent = mecab.morphs(sent)
        text += ' '.join(sent) + '[SEP]'
    return text[:-5]

def sentence_split(text):
    text_list = [sent for sent in map(lambda x: x.strip(), text.split('[SEP]')) if sent != '']
    return text_list

model_checkpoint = 'bert-base-multilingual-cased'
p_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
p_tokenizer.model_max_length = 1536
q_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

class TrainRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, p_tokenizer, q_tokenizer):
        self.dataset = dataset
        self.p_tokenizer = p_tokenizer
        self.q_tokenizer = q_tokenizer

    def __getitem__(self, idx):
        question = self.dataset['question'][idx]
        top20    = self.dataset['top20'][idx]
        target   = self.dataset['answer_idx'][idx]

        p_seqs = self.p_tokenizer(top20,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')

        q_seqs = self.q_tokenizer(question,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')

        p_input_ids = p_seqs['input_ids']
        p_attention_mask = p_seqs['attention_mask']
        p_token_type_ids = p_seqs['token_type_ids']

        q_input_ids = q_seqs['input_ids']
        q_attention_mask = q_seqs['attention_mask']
        q_token_type_ids = q_seqs['token_type_ids']

        p_input_ids_list = torch.Tensor([])
        p_attention_mask_list = torch.Tensor([])
        p_token_type_ids_list = torch.Tensor([])
        for i in range(len(p_attention_mask)):
            str_idx, end_idx = self._select_range(p_attention_mask[i])

            p_input_ids_tmp = torch.cat(
                [torch.Tensor([101]), p_input_ids[i][str_idx:end_idx], torch.Tensor([102])]).int().long()
            p_attention_mask_tmp = p_attention_mask[i][str_idx - 1:end_idx + 1].int().long()
            p_token_type_ids_tmp = p_token_type_ids[i][str_idx - 1:end_idx + 1].int().long()

            p_input_ids_list = torch.cat([p_input_ids_list, p_input_ids_tmp.unsqueeze(0)]).int().long()
            p_attention_mask_list = torch.cat([p_attention_mask_list, p_attention_mask_tmp.unsqueeze(0)])
            p_token_type_ids_list = torch.cat([p_token_type_ids_list, p_token_type_ids_tmp.unsqueeze(0)])

        return p_input_ids_list, p_attention_mask_list, p_token_type_ids_list, q_input_ids, q_attention_mask, q_token_type_ids, target

    def __len__(self):
        return len(self.dataset['question'])

    def _select_range(self, attention_mask):
        sent_len = len([i for i in attention_mask if i != 0])
        if sent_len <= 512:
            return 1, 511
        else:
            start_idx = random.randint(1, sent_len - 511)
            end_idx = start_idx + 510
            return start_idx, end_idx

class TestRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, p_tokenizer, q_tokenizer):
        self.dataset = dataset
        self.p_tokenizer = p_tokenizer
        self.q_tokenizer = q_tokenizer

    def __getitem__(self, idx):
        question = self.dataset['question'][idx]
        top20    = self.dataset['top20'][idx]
        target   = self.dataset['answer_idx'][idx]

        p_seqs = self.p_tokenizer(top20,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')

        q_seqs = self.q_tokenizer(question,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')

        p_input_ids = p_seqs['input_ids']
        p_attention_mask = p_seqs['attention_mask']
        p_token_type_ids = p_seqs['token_type_ids']

        q_input_ids = q_seqs['input_ids']
        q_attention_mask = q_seqs['attention_mask']
        q_token_type_ids = q_seqs['token_type_ids']

        p_input_ids_list = torch.Tensor([])
        p_attention_mask_list = torch.Tensor([])
        p_token_type_ids_list = torch.Tensor([])
        for i in range(len(p_attention_mask)):
            ids_list = self._select_range(p_attention_mask[i])
            if i == target:
                target = list(range(len(p_input_ids_list), len(p_input_ids_list)+len(ids_list)))
            for str_idx, end_idx in ids_list:
                p_input_ids_tmp = torch.cat(
                    [torch.Tensor([101]), p_input_ids[i][str_idx:end_idx], torch.Tensor([102])]).int().long()
                p_attention_mask_tmp = p_attention_mask[i][str_idx - 1:end_idx + 1].int().long()
                p_token_type_ids_tmp = p_token_type_ids[i][str_idx - 1:end_idx + 1].int().long()

                p_input_ids_list = torch.cat([p_input_ids_list, p_input_ids_tmp.unsqueeze(0)]).int().long()
                p_attention_mask_list = torch.cat([p_attention_mask_list, p_attention_mask_tmp.unsqueeze(0)])
                p_token_type_ids_list = torch.cat([p_token_type_ids_list, p_token_type_ids_tmp.unsqueeze(0)])

        return p_input_ids_list, p_attention_mask_list, p_token_type_ids_list, q_input_ids, q_attention_mask, q_token_type_ids, target

    def __len__(self):
        return len(self.dataset['question'])

    def _select_range(self, attention_mask):
        sent_len = len([i for i in attention_mask if i != 0])
        if sent_len <= 512:
            return [(1, 511)]
        else:
            num = sent_len // 255
            res = sent_len % 255
            if res == 0:
                num = -1
            ids_list = []
            for n in range(num):
                if res > 0 and n == num -1:
                    end_idx = sent_len -1
                    start_idx = end_idx - 510
                else:
                    start_idx = n * 255 + 1
                    end_idx = start_idx + 510
                ids_list.append((start_idx, end_idx))
            return ids_list

train_dataset = TrainRetrievalDataset(training_dataset, p_tokenizer, q_tokenizer)
valid_dataset = TestRetrievalDataset(validation_dataset, p_tokenizer, q_tokenizer)

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids= position_ids)
        pooled_output = outputs[1]
        return pooled_output

p_encoder = BertEncoder.from_pretrained(model_checkpoint)
q_encoder = BertEncoder.from_pretrained(model_checkpoint)

if torch.cuda.is_available():
    p_encoder.to('cuda')
    q_encoder.to('cuda')
    print('GPU enabled')

args = TrainingArguments(output_dir='result/dense_retrieval',
                         evaluation_strategy='epoch',
                         learning_rate=1e-5,
                         per_device_train_batch_size=16,
                         per_device_eval_batch_size=1,
                         gradient_accumulation_steps=1,
                         num_train_epochs=10,
                         weight_decay=0.01)


train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=args.per_device_train_batch_size)

valid_sampler = RandomSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset,
                              sampler=train_sampler,
                              batch_size=args.per_device_train_batch_size)

#Optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': args.weight_decay},
    {'params': [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
    {'params': [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': args.weight_decay},
    {'params': [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
scalar = GradScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
criterion = nn.NLLLoss()


# -- logging
log_dir = os.path.join(args.output_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
else:
    raise
logger = SummaryWriter(log_dir=log_dir)

# Start training!
best_loss = 1e9
best_ass = 0.0
global_step = 0


train_iterator = trange(int(args.num_train_epochs), desc='Epoch')
for epoch in train_iterator:
    optimizer.zero_grad()
    p_encoder.zero_grad()
    q_encoder.zero_grad()

    # train
    epoch_iterator = tqdm(train_dataloader, desc='train Iteration')
    p_encoder.to('cuda').train()
    q_encoder.to('cuda').train()
    
    running_loss, running_acc, num_cnt = 0, 0, 0
    with torch.set_grad_enabled(True):
        for step, batch_list in enumerate(epoch_iterator):
            p_input_ids = batch_list[0]
            p_attention_mask = batch_list[1]
            p_token_type_ids = batch_list[2]
            q_input_ids = batch_list[3]
            q_attention_mask = batch_list[4]
            q_token_type_ids = batch_list[5]
            targets_batch = batch_list[6]
            
            for i in range(args.per_device_train_batch_size):
                batch = (p_input_ids[i],
                         p_attention_mask[i],
                         p_token_type_ids[i],
                         q_input_ids[i],
                         q_attention_mask[i],
                         q_token_type_ids[i])
                
                targets = torch.tensor([targets_batch[i]]).long()
                # batch = ty