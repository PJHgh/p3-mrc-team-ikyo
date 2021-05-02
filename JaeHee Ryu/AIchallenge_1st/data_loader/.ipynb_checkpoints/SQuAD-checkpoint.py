# -*- coding: utf-8 -*-
"""
Load SQuAD dataset.
"""

import os
import random
import torch
import numpy as np
import json
from tqdm import tqdm
from codecs import open
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .config import *
from util.file_utils import pickle_dump_large_file, pickle_load_large_file
from konlpy.tag import Mecab

NLP = Mecab()

def word_tokenize(sent):
    # 어절 단위로 mecab에 넣고, 
    # doc = NLP.morphs(sent)
    doc = NLP.pos(sent)
    doc2 = []
    for x in doc :
        w, p = x
        if p[0] == 'J' : w = '~' + w
        doc2.append(w)
    # print(f"MECAB : {sent} -> {doc2}")
    doc = doc2

    return [token for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    # print(f"convert_idx : \n  text = {text}\n  tokens = {tokens}")
    for token in tokens:
        if token[0] == '~' : token = token[1:]
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def filter_func(config, example):
    '''
    if (len(example["context_tokens"]) >= config.para_limit or
            len(example["ques_tokens"]) >= config.ques_limit or
            (example["y2s"][0] - example["y1s"][0]) >= config.ans_limit) :
        print(f'X\tc_tokens = {len(example["context_tokens"])}, q_tokens = {len(example["ques_tokens"])} : Ys =  {example["y2s"]}, {example["y1s"]}')
    '''

    return (len(example["context_tokens"]) >= config.para_limit or
            len(example["ques_tokens"]) >= config.ques_limit or
            (example["y2s"][0] - example["y1s"][0]) >= config.ans_limit)


def get_examples(filename, data_type, word_counter, char_counter, preprocessing=False,
                 debug=False, debug_length=1):
    print("Generating {} examples...".format(data_type))
    examples = []
    meta = {}
    eval_examples = {}

    with open(filename, "r", encoding='utf-8-sig') as fh:
        source = json.load(fh)
        #version = source["version"]
        #meta["version"] = version
        meta["num_q"] = 0
        meta["num_q_answerable"] = 0
        meta["num_qa_answerable"] = 0
        meta["num_q_noanswer"] = 0
        for idx, article in enumerate(tqdm(source["data"])):
            if data_type == 'dev' and idx > 100 : break
            for para in article["paragraphs"] :
                context = para["context"].replace("''", '" ').replace("``", '" ').replace("\u3000", " ")
                if preprocessing :
                    context_tokens = para["ctokens"]
                else :
                    context_tokens = word_tokenize(context)
                    if idx < 10 : print(f"context_tokens = {context_tokens}")

                context_chars = [list(token) for token in context_tokens]
                if idx < 10 : print(f"context_chars = {context_chars}")

                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    meta["num_q"] += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ').replace("\u3000", " ")
                    if preprocessing :
                        ques_tokens = qa["qtokens"]
                    else :
                        ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    
                    if data_type == "test" :
                        y1s, y2s = [], []
                        answer_texts = []
                        answerable = 1
                    else :
                        y1s, y2s = [], []
                        answer_texts = []
                        answers = qa["answers"]
                        answerable = 1
                        #if version == "v2.0" and qa["is_impossible"] is True:
                        #    answers = qa["plausible_answers"]
                        #    answerable = 0
                        meta["num_q_answerable"] += answerable
                        if len(answers) == 0:
                            meta["num_q_noanswer"] += 1
                            continue
                        for answer in answers:
                            answer_text = answer["text"]
                            answer_start = answer['answer_start']
                            answer_end = answer_start + len(answer_text)
                            answer_texts.append(answer_text)
                            answer_span = []
                            for idx, span in enumerate(spans):
                                if not (answer_end <= span[0] or
                                        answer_start >= span[1]):
                                    answer_span.append(idx)
                            #print(answer_text,, answer_start)
                            y1, y2 = answer_span[0], answer_span[-1]
                            y1s.append(y1)
                            y2s.append(y2)
                            meta["num_qa_answerable"] += answerable

                    example = {
                        "context_tokens": context_tokens,
                        "context_chars": context_chars,
                        "ques_tokens": ques_tokens,
                        "ques_chars": ques_chars,
                        "y1s": y1s,
                        "y2s": y2s,
                        "id": meta["num_q"],
                        "context": context,
                        "spans": spans,
                        "answers": answer_texts,
                        "answerable": answerable,
                        "uuid": qa["id"]}
                    examples.append(example)
                    eval_examples[str(meta["num_q"])] = {
                        "context": context,
                        "spans": spans,
                        "answers": answer_texts,
                        "uuid": qa["id"]}
                    if debug and meta["num_q"] >= debug_length:
                        return examples, meta
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, meta, eval_examples


def get_embedding(counter, data_type,
                  emb_file=None, size=None, vec_size=None,
                  limit=-1, specials=["<PAD>", "<OOV>", "<SOS>", "<EOS>"]):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    # filtered_elements = [k for k, v in counter.items() if v > limit] # Freq로 정하기
    # dykim : vocab의 갯수를 지정하기
    filtered_elements=list()
    counter_sort = sorted(counter.items(), key=(lambda x:x[1]), reverse=True)
    for idx, (k, v) in enumerate(counter_sort) :
        if idx >= limit-4 :
            break
        filtered_elements.append(k)
    # end dykim

    print(f"get_embedding : ({len(filtered_elements)}) = {filtered_elements[:30]} . . . ")

    assert vec_size is not None
    for token in filtered_elements:
        embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
    print("{} tokens have corresponding embedding vector".format(len(filtered_elements)))

    token2idx_dict = {token: idx
                      for idx, token
                      in enumerate(embedding_dict.keys(), len(specials))}
    for i in range(len(specials)):
        token2idx_dict[specials[i]] = i
        embedding_dict[specials[i]] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def word2wid(word, word2idx_dict):
    for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in word2idx_dict:
            return word2idx_dict[each]
    return word2idx_dict["<OOV>"]


def char2cid(char, char2idx_dict):
    if char in char2idx_dict:
        return char2idx_dict[char]
    return char2idx_dict["<OOV>"]


def save(filepath, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    pickle_dump_large_file(obj, filepath)


def build_features(config, examples, meta, data_type,
                   word2idx_dict, char2idx_dict, debug=False):
    print("Processing {} examples...".format(data_type))
    total = 0
    total_ = 0
    examples_with_features = []
    for example in tqdm(examples):
        total_ += 1
        if data_type != 'test' and filter_func(config, example) :
            continue
        total += 1

        if data_type == 'test' :
            context_wids = np.ones(
                [2400], dtype=np.int32) * \
                word2idx_dict["<PAD>"]
            context_cids = np.ones(
                [2400, config.char_limit], dtype=np.int32) * \
                char2idx_dict["<PAD>"]
            question_wids = np.ones([config.ques_limit], dtype=np.int32) * \
                word2idx_dict["<PAD>"]
            question_cids = np.ones(
                [config.ques_limit, config.char_limit], dtype=np.int32) * \
                char2idx_dict["<PAD>"]
            y1 = np.zeros([2400], dtype=np.float32)
            y2 = np.zeros([2400], dtype=np.float32)

            for i, token in enumerate(example["context_tokens"]):
                context_wids[i] = word2wid(token, word2idx_dict)

            for i, token in enumerate(example["ques_tokens"]):
                question_wids[i] = word2wid(token, word2idx_dict)

            for i, token in enumerate(example["context_chars"]):
                for j, char in enumerate(token):
                    if j == config.char_limit:
                        break
                    context_cids[i, j] = char2cid(char, char2idx_dict)

        else :
            context_wids = np.ones(
                [config.para_limit], dtype=np.int32) * \
                word2idx_dict["<PAD>"]
            context_cids = np.ones(
                [config.para_limit, config.char_limit], dtype=np.int32) * \
                char2idx_dict["<PAD>"]
            question_wids = np.ones([config.ques_limit], dtype=np.int32) * \
                word2idx_dict["<PAD>"]
            question_cids = np.ones(
                [config.ques_limit, config.char_limit], dtype=np.int32) * \
                char2idx_dict["<PAD>"]
            y1 = np.zeros([config.para_limit], dtype=np.float32)
            y2 = np.zeros([config.para_limit], dtype=np.float32)

            for i, token in enumerate(example["context_tokens"]):
                if i >= config.para_limit :
                    # print(f'ERROR : context_tokens len = {len(example["context_tokens"])} : {example["context_tokens"]}')
                    continue
                context_wids[i] = word2wid(token, word2idx_dict)

            for i, token in enumerate(example["ques_tokens"]):
                question_wids[i] = word2wid(token, word2idx_dict)

            for i, token in enumerate(example["context_chars"]):
                if i >= config.para_limit :
                    continue
                for j, char in enumerate(token):
                    if j == config.char_limit:
                        break
                    context_cids[i, j] = char2cid(char, char2idx_dict)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == config.char_limit:
                    break
                question_cids[i, j] = char2cid(char, char2idx_dict)

        if data_type == 'test' :
            example["context_wids"] = context_wids
            example["context_cids"] = context_cids
            example["question_wids"] = question_wids
            example["question_cids"] = question_cids
            example["y1"] = 0
            example["y2"] = 0

        else :
            # !!! use last answer as the target answer
            start, end = example["y1s"][-1], example["y2s"][-1]
            y1[start], y2[end] = 1.0, 1.0

            example["context_wids"] = context_wids
            example["context_cids"] = context_cids
            example["question_wids"] = question_wids
            example["question_cids"] = question_cids
            example["y1"] = start
            example["y2"] = end

            # don't store unnecessary properties
            # to save shared memory when loading data
            example["spans"] = None
            example["context_tokens"] = None
            example["context_chars"] = None
            example["ques_tokens"] = None
            example["ques_chars"] = None
        
        examples_with_features.append(example)

    print("Built {} / {} instances of features in total".format(total, total_))
    meta["num_q_filtered"] = total
    return examples_with_features, meta

def prepro_dev(config):
    dev_examples, dev_meta, dev_eval = get_examples(config.dev_file, "dev", word_counter, char_counter, config.preprocessing)
    save(config.dev_eval_file, dev_eval, message="dev eval")
    del dev_eval

    dev_examples, dev_meta = build_features(
    config, dev_examples, dev_meta, "dev",
    word2idx_dict, char2idx_dict)
    save(config.dev_examples_file, dev_examples, message="dev examples")
    del dev_examples
    save(config.dev_meta_file, dev_meta, message="dev meta")
    del dev_meta


def prepro(config):
    word_counter, char_counter = Counter(), Counter()

    train_examples, train_meta, train_eval = get_examples(config.train_file, "train", word_counter, char_counter, config.preprocessing)
    save(config.train_eval_file, train_eval, message="train eval")
    del train_eval

    dev_examples, dev_meta, dev_eval = get_examples(config.dev_file, "dev", word_counter, char_counter, config.preprocessing)
    save(config.dev_eval_file, dev_eval, message="dev eval")
    del dev_eval

    test_examples, test_meta, _ = get_examples(config.test_file, "test", word_counter, char_counter, config.preprocessing)

    word_emb_file = config.glove_word_file
    word_emb_size = config.glove_word_size
    word_emb_dim = config.glove_dim
    pretrained_char = config.pretrained_char
    char_emb_file = config.glove_char_file if pretrained_char else None
    char_emb_size = config.glove_char_size if pretrained_char else None
    char_emb_dim = config.glove_dim if pretrained_char else config.char_dim

    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file,
        size=word_emb_size, vec_size=word_emb_dim, limit=30000)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file,
        size=char_emb_size, vec_size=char_emb_dim, limit=1000)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")
    del word_emb_mat
    del char_emb_mat

    train_examples, train_meta = build_features(
        config, train_examples, train_meta, "train",
        word2idx_dict, char2idx_dict)
    save(config.train_meta_file, train_meta, message="train meta")
    del train_meta
    save(config.train_examples_file, train_examples, message="train examples")
    del train_examples
    
    dev_examples, dev_meta = build_features(
        config, dev_examples, dev_meta, "dev",
        word2idx_dict, char2idx_dict)
    save(config.dev_examples_file, dev_examples, message="dev examples")
    del dev_examples
    save(config.dev_meta_file, dev_meta, message="dev meta")
    del dev_meta

    word2idx_dict = pickle_load_large_file(config.word_dictionary)
    char2idx_dict = pickle_load_large_file(config.char_dictionary)

    test_examples, test_meta = build_features(
        config, test_examples, test_meta, "test",
        word2idx_dict, char2idx_dict)
    del test_meta
    save(config.test_examples_file, test_examples, message="test examples")
    del test_examples

    del word2idx_dict
    del char2idx_dict


class SQuAD(Dataset):
    def __init__(self, examples_file, mode):
        self.mode = mode
        self.examples = pickle_load_large_file(examples_file)
        self.num = len(self.examples)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.mode == 'test' :
            return (self.examples[idx]["context_wids"],
                self.examples[idx]["context_cids"],
                self.examples[idx]["question_wids"],
                self.examples[idx]["question_cids"],
                self.examples[idx]["y1"],
                self.examples[idx]["y2"],
                self.examples[idx]["y1s"],
                self.examples[idx]["y2s"],
                self.examples[idx]["id"],
                self.examples[idx]["answerable"],
                self.examples[idx]["context_tokens"],
                self.examples[idx]["uuid"])
        else :
            return (self.examples[idx]["context_wids"],
                self.examples[idx]["context_cids"],
                self.examples[idx]["question_wids"],
                self.examples[idx]["question_cids"],
                self.examples[idx]["y1"],
                self.examples[idx]["y2"],
                self.examples[idx]["y1s"],
                self.examples[idx]["y2s"],
                self.examples[idx]["id"],
                self.examples[idx]["answerable"])

def collate(data):
    Cwid, Ccid, Qwid, Qcid, y1, y2, y1s, y2s, id, answerable = zip(*data)
    Cwid = torch.tensor(Cwid).long()
    Ccid = torch.tensor(Ccid).long()
    Qwid = torch.tensor(Qwid).long()
    Qcid = torch.tensor(Qcid).long()
    y1 = torch.from_numpy(np.array(y1)).long()
    y2 = torch.from_numpy(np.array(y2)).long()
    id = torch.from_numpy(np.array(id)).long()
    answerable = torch.tensor(answerable).long()
    return Cwid, Ccid, Qwid, Qcid, y1, y2, y1s, y2s, id, answerable

def collate_test(data):
    Cwid, Ccid, Qwid, Qcid, y1, y2, y1s, y2s, id, answerable, context_tok, uuid = zip(*data)
    Cwid = torch.tensor(Cwid).long()
    Ccid = torch.tensor(Ccid).long()
    Qwid = torch.tensor(Qwid).long()
    Qcid = torch.tensor(Qcid).long()
    y1 = torch.from_numpy(np.array(y1)).long()
    y2 = torch.from_numpy(np.array(y2)).long()
    id = torch.from_numpy(np.array(id)).long()
    answerable = torch.tensor(answerable).long()
    return Cwid, Ccid, Qwid, Qcid, y1, y2, y1s, y2s, id, answerable, context_tok, uuid

def get_loader(mode, examples_file, batch_size, shuffle=True):
    dataset = SQuAD(examples_file, mode)
    print(examples_file, dataset.num)
    
    if mode == 'train' :
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # num_works > 0 may cause RequestRefused error
            collate_fn=collate)
    else :
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # num_works > 0 may cause RequestRefused error
            collate_fn=collate_test)
    return data_loader
