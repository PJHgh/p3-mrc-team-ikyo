# -*- coding: utf-8 -*-
"""
Main file for training SQuAD reading comprehension model.
"""
import os
import sys
import json
import argparse
import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from data_loader.SQuAD import prepro, get_loader
from model.QANet import QANet
from trainer.QANet_trainer import Trainer
from model.modules.ema import EMA
from util.file_utils import pickle_load_large_file


data_folder = "dataset/"
pre_folder = "2400_140_mecab/"
parser = argparse.ArgumentParser(description='Lucy')

# mode
parser.add_argument(
    '--mode',
    default='train', 
    type=str, help='mode of models (train, test)')

# dataset
parser.add_argument(
    '--processed_data',
    default=False, action='store_true',
    help='whether the dataset already processed')
parser.add_argument(
    '--train_file',
    default=data_folder + 'train.json',
    type=str, help='path of train dataset')
parser.add_argument(
    '--dev_file',
    default=data_folder + 'validate.json',
    type=str, help='path of dev dataset')
parser.add_argument(
    '--test_file',
    default=data_folder + 'test.json',
    type=str, help='path of test dataset')
parser.add_argument(
    '--train_examples_file',
    default=data_folder + pre_folder + 'train-v1.1-examples.pkl',
    type=str, help='path of train dataset examples file')
parser.add_argument(
    '--dev_examples_file',
    default=data_folder + pre_folder + 'valiate-v1.1-examples.pkl',
    type=str, help='path of dev dataset examples file')
parser.add_argument(
    '--test_examples_file',
    default=data_folder + pre_folder + 'test-v1.1-examples.pkl',
    type=str, help='path of test dataset examples file')
parser.add_argument(
    '--train_meta_file',
    default=data_folder + pre_folder + 'train-v1.1-meta.pkl',
    type=str, help='path of train dataset meta file')
parser.add_argument(
    '--dev_meta_file',
    default=data_folder + pre_folder + 'validate-v1.1-meta.pkl',
    type=str, help='path of dev dataset meta file')
parser.add_argument(
    '--train_eval_file',
    default=data_folder + pre_folder + 'train-v1.1-eval.pkl',
    type=str, help='path of train dataset eval file')
parser.add_argument(
    '--dev_eval_file',
    default=data_folder + pre_folder + 'validate-v1.1-eval.pkl',
    type=str, help='path of dev dataset eval file')
parser.add_argument(
    '--val_num_batches',
    default=500, type=int,
    help='number of batches for evaluation (default: 500)')

# embedding
parser.add_argument(
    '--glove_word_file',
    #default=data_folder + 'glove.840B.300d.txt',
    default=None,
    type=str, help='path of word embedding file')
parser.add_argument(
    '--glove_word_size',
    default=int(2.2e6), type=int,
    help='Corpus size for Glove')
parser.add_argument(
    '--glove_dim',
    default=300, type=int,
    help='word embedding size (default: 300)')
parser.add_argument(
    '--word_emb_file',
    default=data_folder + pre_folder + 'word_emb.pkl',
    type=str, help='path of word embedding matrix file')
parser.add_argument(
    '--word_dictionary',
    default=data_folder + pre_folder + 'word_dict.pkl',
    type=str, help='path of word embedding dict file')

parser.add_argument(
    '--pretrained_char',
    default=False, action='store_true',
    help='whether train char embedding or not')
parser.add_argument(
    '--glove_char_file',
    #default=data_folder + "glove.840B.300d-char.txt",
    default=None,
    type=str, help='path of char embedding file')
parser.add_argument(
    '--glove_char_size',
    default=94, type=int,
    help='Corpus size for char embedding')
parser.add_argument(
    '--char_dim',
    default=64, type=int,
    help='char embedding size (default: 64)')
parser.add_argument(
    '--char_emb_file',
    default=data_folder + pre_folder + 'char_emb.pkl',
    type=str, help='path of char embedding matrix file')
parser.add_argument(
    '--char_dictionary',
    default=data_folder + pre_folder + 'char_dict.pkl',
    type=str, help='path of char embedding dict file')
parser.add_argument(
    '--preprocessing',
    default=False, action='store_true',
    help='whether tokens include in dataset')

# train
parser.add_argument(
    '-b', '--batch_size',
    default=20, type=int,
    help='mini-batch size (default: 32)')
parser.add_argument(
    '-e', '--epochs',
    default=30, type=int,
    help='number of total epochs (default: 30)')

# debug
parser.add_argument(
    '--debug',
    default=False, action='store_true',
    help='debug mode or not')
parser.add_argument(
    '--debug_batchnum',
    default=2, type=int,
    help='only train and test a few batches when debug (devault: 2)')

# checkpoint
parser.add_argument(
    '--resume',
    default='', type=str,
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--verbosity',
    default=2, type=int,
    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument(
    '--save_dir',
    default='output/', type=str,
    help='directory of saved model (default: checkpoints/)')
parser.add_argument(
    '--save_freq',
    default=1, type=int,
    help='training checkpoint frequency (default: 1 epoch)')
parser.add_argument(
    '--print_freq',
    default=1, type=int,
    help='print training information frequency (default: 10 steps)')

# cuda
parser.add_argument(
    '--with_cuda',
    default=True, action='store_true',
    help='use CPU in case there\'s no GPU support')
parser.add_argument(
    '--multi_gpu',
    default=False, action='store_true',
    help='use multi-GPU in case there\'s multiple GPUs available')

# log & visualize
parser.add_argument(
    '--visualizer',
    default=False, action='store_true',
    help='use visdom visualizer or not')
parser.add_argument(
    '--log_file',
    default='log.txt',
    type=str, help='path of log file')

# optimizer & scheduler & weight & exponential moving average
parser.add_argument(
    '--lr',
    default=0.001, type=float,
    help='learning rate')
parser.add_argument(
    '--lr_warm_up_num',
    default=1000, type=int,
    help='number of warm-up steps of learning rate')
parser.add_argument(
    '--beta1',
    default=0.8, type=float,
    help='beta 1')
parser.add_argument(
    '--beta2',
    default=0.999, type=float,
    help='beta 2')
parser.add_argument(
    '--decay',
    default=0.9999, type=float,
    help='exponential moving average decay')
parser.add_argument(
    '--use_scheduler',
    default=True, action='store_false',
    help='whether use learning rate scheduler')
parser.add_argument(
    '--use_grad_clip',
    default=True, action='store_false',
    help='whether use gradient clip')
parser.add_argument(
    '--grad_clip',
    default=5.0, type=float,
    help='global Norm gradient clipping rate')
parser.add_argument(
    '--use_ema',
    default=True, action='store_true',
    help='whethere use exponential moving average')
parser.add_argument(
    '--use_early_stop',
    default=False, action='store_false',
    help='whether use early stop')
parser.add_argument(
    '--early_stop',
    default=10, type=int,
    help='checkpoints for early stop')

# model
parser.add_argument(
    '--para_limit',
    default=720, type=int,  # 2400
    help='maximum context token number')
parser.add_argument(
    '--ques_limit',
    default=140, type=int,
    help='maximum question token number')
parser.add_argument(
    '--ans_limit',
    default=30, type=int,   # 100
    help='maximum answer token number')
parser.add_argument(
    '--char_limit',
    default=8, type=int,
    help='maximum char number in a word')
parser.add_argument(
    '--d_model',
    default=128, type=int,
    help='model hidden size')
parser.add_argument(
    '--num_head',
    default=8, type=int,
    help='attention num head')


def convert_tokens(uuids, tokens, pp1, pp2):
    answer_dict = {}
    for uuid, tok, p1, p2 in zip(uuids, tokens, pp1, pp2):
        answer_text = " ".join(tok[p1:p2+1])
        answer_dict[uuid] = answer_text
    #print(answer_dict)
    return answer_dict

def test(model, data_loader, args, device):
    model.eval()
    answer_dict = {}
  
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            (context_wids,
                context_cids,
                question_wids,
                question_cids,
                y1,
                y2,
                y1s,
                y2s,
                id,
                answerable,
                context_tok,
                uuids) = batch

            question_wids = question_wids.to(device)
            question_cids = question_cids.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            answerable = answerable.to(device)

            window_size = 500
            context_length = len(context_wids[0])
            ymin_list = [0] * len(context_wids)
            ymax_list = [0] * len(context_wids)
            max_score = [0] * len(context_wids)
            context_list = [''] * len(context_wids)

            for sub_len in range(0, context_length, window_size) :
                context_subs_word = list()
                context_subs_char = list()

                for idx in range(len(context_wids)) :
                    context_word = context_wids[idx][sub_len:sub_len+args.para_limit].tolist()
                    context_char = context_cids[idx][sub_len:sub_len+args.para_limit].tolist()

                    if len(context_word) <= 50 :
                        continue

                    context_subs_word.append(context_word)
                    context_subs_char.append(context_char)

                context_subs_word = torch.as_tensor(context_subs_word)
                context_subs_char = torch.as_tensor(context_subs_char)

                context_subs_word = context_subs_word.to(device)
                context_subs_char = context_subs_char.to(device)

                p1, p2 = model(
                        context_subs_word,
                        context_subs_char,
                        question_wids,
                        question_cids)

                s1, pos1 = torch.max(p1, dim=1)
                s2, pos2 = torch.max(p2, dim=1)

                s1 = s1.tolist()
                s2 = s2.tolist()
                pos1 = pos1.tolist()
                pos2 = pos2.tolist()

                idx = 0
                pre_idx = -1
                while idx < len(context_wids) :
                    if sub_len == 0 :
                        p1 = F.softmax(p1, dim=1)
                        p2 = F.softmax(p2, dim=1)
                        outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
                        for j in range(outer.size()[0]):
                            outer[j] = torch.triu(outer[j])
                            # outer[j] = torch.tril(outer[j], self.args.ans_limit)
                        a1, _ = torch.max(outer, dim=2)
                        a2, _ = torch.max(outer, dim=1)
                        ymin = torch.argmax(a1, dim=1)
                        ymax = torch.argmax(a2, dim=1)

                        max_score[idx] = s1[idx]+s2[idx]
                        ymin_list[idx] = ymin[idx]
                        ymax_list[idx] = ymax[idx]
                        context_list[idx] = context_tok[idx][sub_len:sub_len+args.para_limit]

                    if s1[idx]+s2[idx] > max_score[idx] :
                        if pos1[idx] > pos2[idx] :
                            idx += 1
                            continue
                       
                        max_score[idx] = s1[idx]+s2[idx]
                        ymin_list[idx] = pos1[idx]
                        ymax_list[idx] = pos2[idx]
                        context_list[idx] = context_tok[idx][sub_len:sub_len+args.para_limit]
                    idx += 1

            answer_dict_ = convert_tokens(uuids, context_list, ymin_list, ymax_list)
            for id in answer_dict_ :
                answer_dict_[id] = answer_dict_[id].replace("~", "")
            answer_dict.update(answer_dict_)

    with open(args.save_dir + "predictions.json", "w", encoding="utf-8-sig") as wf :
        json.dump(answer_dict, wf)


def test_entry(model, args, device):
    test_dataset = get_loader(args.mode, args.test_examples_file, args.batch_size, shuffle=False)
    print(f"TEST SET LOADING : {len(test_dataset)} Done")
    test(model, test_dataset, args, device)

def main(args):
    # show configuration
    print(args)
    random_seed = None

    # set log file
    log = sys.stdout
    if args.log_file is not None:
        log = open(args.log_file, "a")

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu")

    # process word vectors and datasets
    if not args.processed_data:
        prepro(args)

    # load word vectors and datasets
    wv_tensor = torch.FloatTensor(np.array(pickle_load_large_file(args.word_emb_file), dtype=np.float32))
    cv_tensor = torch.FloatTensor(np.array(pickle_load_large_file(args.char_emb_file), dtype=np.float32))
    wv_word2ix = pickle_load_large_file(args.word_dictionary)

    # construct model
    model = QANet(
        wv_tensor,
        cv_tensor,
        args.para_limit,
        args.ques_limit,
        args.d_model,
        num_head=args.num_head,
        train_cemb=(not args.pretrained_char),
        pad=wv_word2ix["<PAD>"])
    model.summary()

    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    if args.mode == 'train' :

        train_dataloader = get_loader(
            args.mode, args.train_examples_file, args.batch_size, shuffle=True)
        dev_dataloader = get_loader(
            args.mode, args.dev_examples_file, args.batch_size, shuffle=True)

        print('Number of Train Dataset :', len(train_dataloader))
        print('Number of Validate Dataset :', len(dev_dataloader))

        # exponential moving average
        ema = EMA(args.decay)
        if args.use_ema:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.register(name, param.data)

        # set optimizer and scheduler
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(
            params=parameters,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=1e-8,
            weight_decay=3e-7)
        cr = 1.0 / math.log(args.lr_warm_up_num)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda ee: cr * math.log(ee + 1)
            if ee < args.lr_warm_up_num else 1)

        # set loss, metrics
        loss = torch.nn.CrossEntropyLoss()

        # set visdom visualizer to store training process information
        # see the training process on http://localhost:8097/
        vis = None

        # construct trainer
        # an identifier (prefix) for saved model
        identifier = type(model).__name__ + '_'
        trainer = Trainer(
            args, model, loss,
            train_data_loader=train_dataloader,
            dev_data_loader=dev_dataloader,
            train_eval_file=args.train_eval_file,
            dev_eval_file=args.dev_eval_file,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            with_cuda=args.with_cuda,
            save_dir=args.save_dir,
            verbosity=args.verbosity,
            save_freq=args.save_freq,
            print_freq=args.print_freq,
            resume=args.resume,
            identifier=identifier,
            debug=args.debug,
            debug_batchnum=args.debug_batchnum,
            lr=args.lr,
            lr_warm_up_num=args.lr_warm_up_num,
            grad_clip=args.grad_clip,
            decay=args.decay,
            visualizer=vis,
            logger=log,
            use_scheduler=args.use_scheduler,
            use_grad_clip=args.use_grad_clip,
            use_ema=args.use_ema,
            ema=ema,
            use_early_stop=args.use_early_stop,
            early_stop=args.early_stop)

        # start training!
        start = datetime.now()
        trainer.train()
        print("Time of training model ", datetime.now() - start)

    if args.mode == 'eval' :
        dev_dataloader = get_loader(args.mode, args.dev_examples_file, args.batch_size, shuffle=True)

        print('Number of Validate Dataset :', len(dev_dataloader))

        # exponential moving average
        ema = EMA(args.decay)
        if args.use_ema:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.register(name, param.data)

        # set optimizer and scheduler
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(
            params=parameters,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=1e-8,
            weight_decay=3e-7)
        cr = 1.0 / math.log(args.lr_warm_up_num)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda ee: cr * math.log(ee + 1)
            if ee < args.lr_warm_up_num else 1)

        # set loss, metrics
        loss = torch.nn.CrossEntropyLoss()

        # set visdom visualizer to store training process information
        # see the training process on http://localhost:8097/
        vis = None

        # construct trainer
        # an identifier (prefix) for saved model
        identifier = type(model).__name__ + '_'
        trainer = Trainer(
            args, model, loss,
            train_data_loader=train_dataloader,
            dev_data_loader=dev_dataloader,
            train_eval_file=args.train_eval_file,
            dev_eval_file=args.dev_eval_file,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            with_cuda=args.with_cuda,
            save_dir=args.save_dir,
            verbosity=args.verbosity,
            save_freq=args.save_freq,
            print_freq=args.print_freq,
            resume=args.resume,
            identifier=identifier,
            debug=args.debug,
            debug_batchnum=args.debug_batchnum,
            lr=args.lr,
            lr_warm_up_num=args.lr_warm_up_num,
            grad_clip=args.grad_clip,
            decay=args.decay,
            visualizer=vis,
            logger=log,
            use_scheduler=args.use_scheduler,
            use_grad_clip=args.use_grad_clip,
            use_ema=args.use_ema,
            ema=ema,
            use_early_stop=args.use_early_stop,
            early_stop=args.early_stop)

        # start training!
        start = datetime.now()
        trainer.train()
        print("Time of training model ", datetime.now() - start)


    elif args.mode == 'test' :
        print("Loading checkpoint: {} ...".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        test_entry(model, args, device)


if __name__ == '__main__':
    main(parser.parse_args())
