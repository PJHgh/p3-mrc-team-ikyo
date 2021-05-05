import logging
import os
import sys
import time
import json

import torch
import random
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AdamW
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from konlpy.tag import Mecab

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from elasticsearch_retrieval import *
from data_processing import DataProcessor
from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval
from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

# config 설정
def get_config():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args

# seed 설정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

# 모델, tokenizer, config 가져오기
def get_model(model_args, training_args):
    # MRC 모델을 저장할 때 backborn 모델이름으로 저장해서 사용했기 때문
    # 바꾸는게 좋아보이긴 하는데...
    backborn_name = model_args.model_name_or_path.split("/")[-1][:-3]
    if backborn_name == "koelectra-base-v3-discriminator":
        backborn_name = "monologg/koelectra-base-v3-discriminator"

    tokenizer = AutoTokenizer.from_pretrained(
        backborn_name,
        use_fast=True
    )
    model = torch.load(model_args.model_name_or_path)

    return tokenizer, model


def run_sparse_retrieval(datasets, training_args):
    retriever = SparseRetrieval(tokenize_fn=tokenize,
                                data_path="/opt/ml/input/data/data/",
                                context_path="wikipedia_documents.json")

    retriever.get_sparse_embedding()
    # validation set에 대한 예측문장을 가져온다.
    df = retriever.retrieve(datasets['validation'])

    # faiss retrieval 사용하고싶으면 사용
    # df = retriever.retrieve_faiss(dataset['validation'])

    if training_args.do_predict: # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
        f = Features({'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})

    elif training_args.do_eval: # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
        f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None),
                      'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})

    # 예측문장과 질문을 Dataset으로
    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})
    return datasets

# dense retrival 사용 함수 (킹종헌님꺼 쓸거 같으니까 생략할께요)
def dpr(text_data):
    with open("/opt/ml/input/data/data/wikipedia_documents.json", "r") as f:
        wiki = json.load(f)
    contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))

    q_model = torch.load("/opt/ml/lastcode/retrieval_model/q_model.pt")
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    wiki_embs = torch.load("/opt/ml/lastcode/retrieval_model/wiki_embs.pt")

    test_data = text_data["validation"]["question"]
    test_token = q_tokenizer(test_data, padding=True, truncation=True, max_length=100, return_tensors='pt')
    test_dataset = TensorDataset(test_token["input_ids"], test_token["attention_mask"], test_token["token_type_ids"])
    test_iter = DataLoader(test_dataset, batch_size=1)

    with torch.no_grad():
        q_model.eval()
        pbar = tqdm(enumerate(test_iter), total=len(test_iter), position=0, leave=True)
        total = []
        for step, batch in pbar:
            batch = tuple(t.cuda() for t in batch)

            q_inputs = { 
                "input_ids" : batch[0],
                "attention_mask" : batch[1],
                "token_type_ids" : batch[2]
            }

            q_emb = q_model(**q_inputs).pooler_output.to("cpu")
            dot_prod_scores = torch.matmul(q_emb, torch.transpose(wiki_embs, 0, 1))
            rank = torch.argsort(dot_prod_scores, dim=1 ,descending=True).squeeze()

            tmp = {
                "question" : text_data["validation"]["question"][step],
                "id" : text_data["validation"]["id"][step],
                "context" : contexts[rank[0]]
            }
            total.append(tmp)
    
    df = pd.DataFrame(total)
    f = Features({'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})
    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})

    return datasets

# elastic search
def run_elastic_retrival(text_data):
    es, index_name = elastic_setting()
    question_texts = text_data["validation"]["question"]
    total = []
    scores = []
    n_results = 1

    # question을 돌면서 context를 뽑아온다.
    pbar = tqdm(enumerate(question_texts), total=len(question_texts), position=0, leave=True)
    for step, question_text in pbar:
        context_list = elastic_retrieval(es, index_name, question_text, n_results)
        # score가 처음 시도했던 방식에 필요해서 만든 건대 무시하셔도 됩니다.
        score = []
        for i in range(len(context_list)):
            # 질문과 질문으로 elastic search에서 얻은 context를 같이 사용할 수 있도록 만들어준다.
            tmp = {
                "question" : question_text,
                "id" : text_data["validation"]["id"][step] + f"_{i}",
                "context" : context_list[i][0]
            }
            score.append(context_list[i][1])
            # 질문과 context가 있는 tmp들을 append
            total.append(tmp)
        scores.append(score)

    df = pd.DataFrame(total)
    f = Features({'context': Value(dtype='string', id=None),
                'id': Value(dtype='string', id=None),
                'question': Value(dtype='string', id=None)})
    
    
    # 질문과, context를 dataset으로 변환 후 return
    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})

    return datasets, scores

# run_elastic_retrival와 거의 유사합니다. 
def run_concat_elastic_retrival(text_data):
    es, index_name = elastic_setting()
    question_texts = text_data["validation"]["question"]
    total = []
    scores = []
    n_results = 7

    pbar = tqdm(enumerate(question_texts), total=len(question_texts), position=0, leave=True)
    for step, question_text in pbar:
        context_list = elastic_retrieval(es, index_name, question_text, n_results)
        score = []
        concat_context = ""
        # 유일하게 다른 부분 : context list를 concat 시켜주는 부분
        for i in range(len(context_list)):
            if i == 0 :
                concat_context += context_list[i][0]
            else:
                concat_context += " " + context_list[i][0]

        tmp = {
            "question" : question_text,
            "id" : text_data["validation"]["id"][step] + "_0",
            "context" : concat_context
        }

        score.append(context_list[0][1])
        total.append(tmp)
        scores.append(score)

    df = pd.DataFrame(total)
    f = Features({'context': Value(dtype='string', id=None),
                'id': Value(dtype='string', id=None),
                'question': Value(dtype='string', id=None)})
    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})

    return datasets, scores


def get_data(training_args, tokenizer, text_data_path = "/opt/ml/input/data/data/test_dataset"):
    text_data = load_from_disk(text_data_path)
    
    # 사용하고 싶은 retrieval 선택하여 사용 (4개중 1개)
    #text_data = run_sparse_retrieval(text_data, training_args) #sparse_retrieval
    #text_data = dpr(text_data) #dense retrieval
    #text_data, scores = run_elastic_retrival(text_data) #elasticsearch retrival
    text_data, scores = run_concat_elastic_retrival(text_data) #elasticsearch retrival concat

    column_names = text_data["validation"].column_names

    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )
    # 데이터 tokenize(mrc 모델안에 들어 갈 수 있도록)
    data_processor = DataProcessor(tokenizer)
    val_text = text_data["validation"]
    val_dataset = data_processor.val_tokenzier(val_text, column_names)
    val_iter = DataLoader(val_dataset, collate_fn = data_collator, batch_size=1)

    return text_data, val_iter, val_dataset, scores

# baseline과 같으니 생략
def post_processing_function(examples, features, predictions, text_data, data_args, training_args):
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=data_args.max_answer_length,
        output_dir=training_args.output_dir,
    )

    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    if training_args.do_predict:
        return formatted_predictions

    elif training_args.do_eval:
        references = [
            {"id": ex["id"], "answers": ex["answers"]}
            for ex in text_data["validation"]
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

# baseline과 같으니 생략
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    step = 0

    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)

    for i, output_logit in enumerate(start_or_end_logits):
        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat

# MRC를 이용하여 정답 예측 prediction json 생성
def predict(model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device):
    metric = load_metric("squad")
    # xlm의 input 예외처리
    if "xlm" in model_args.model_name_or_path:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids"])
    else:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])

    model.eval()

    all_start_logits = []
    all_end_logits = []

    t = time.time()
    # 예측 시작
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
    for step, batch in pbar:
        batch = batch.to(device)
        outputs = model(**batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        all_start_logits.append(start_logits.detach().cpu().numpy())
        all_end_logits.append(end_logits.detach().cpu().numpy())
    
    max_len = max(x.shape[1] for x in all_start_logits)

    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)

    del all_start_logits
    del all_end_logits
    
    test_dataset.set_format(type=None, columns=list(test_dataset.features.keys()))
    output_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(text_data["validation"], test_dataset, output_numpy, text_data, data_args, training_args)

# preiction.json을 이용하여 최종 예측결과 전처리 (EM score를 높이기 위하여)
def make_submission(scores, training_args):
    macab = Mecab()
    with open("/opt/ml/lastcode/submission/nbest_predictions.json", "r") as f:
        nbest = json.load(f)

    prediction = dict()
    prev_mrc_id = None
    # 역시 score는 무시해주세요
    final_score = []
    score_step = 0
    # 마지막에 주로 붙었던 조사들로 이루어진 set
    last_word = {"은", "는", "이" ,"가", "을" ,"를", "의", "에"}

    for mrc_id_step in nbest.keys():
        # score와 관련된 split
        mrc_id, step = mrc_id_step.split("_")
        if prev_mrc_id != mrc_id:
            if prev_mrc_id is not None:
                large_step = np.argmax(final_score)
                final_predictions = nbest[prev_mrc_id+f"_{large_step}"][0]["text"]
                pos_tag = macab.pos(final_predictions)
                # last word(조사)에 있는 단어고 형태소 분석 결과가 j일경우 삭제
                if pos_tag[-1][1][0] == "J" and final_predictions[-1] in last_word:
                    final_predictions = final_predictions[:-1]

                prediction[str(prev_mrc_id)] = final_predictions
                score_step += 1
                final_score = []
            sum_score = sum(scores[score_step])
            prev_mrc_id = mrc_id
        
        # 여기다가 (1 - m) (m) 안쓰는것...
        final_score.append(nbest[mrc_id_step][0]["probability"] + scores[score_step][int(step)]/sum_score)
    
    # 전처리한 최종결과 final_prediction으로 저장
    with open(training_args.output_dir+f"/final_predictions.json", 'w', encoding='utf-8') as make_file:
        json.dump(prediction, make_file, indent="\t", ensure_ascii=False)
    print(prediction)



def main():
    # args 가져오기
    model_args, data_args, training_args = get_config()
    # seed 설정
    seed_everything(training_args.seed)
    # device
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tokenizer, model 가져오기
    tokenizer, model  = get_model(model_args, training_args)
    model.cuda()
    # data 가져오기
    text_data, test_loader, test_dataset, scores = get_data(training_args, tokenizer)
    # prediction.json 생성
    predict(model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device)
    # 최종 final_prediction 생성 => 제출
    make_submission(scores, training_args)

if __name__ == "__main__":
    main()