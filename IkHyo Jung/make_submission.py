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

def get_model(model_args, training_args):
    backborn_name = model_args.model_name_or_path.split("/")[-1][:-3]
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

# dense retrival 사용 함수
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

def run_elastic_retrival(text_data):
    es, index_name = elastic_setting()
    question_texts = text_data["validation"]["question"]
    total = []
    scores = []
    n_results = 5

    pbar = tqdm(enumerate(question_texts), total=len(question_texts), position=0, leave=True)
    for step, question_text in pbar:
        context_list = elastic_retrieval(es, index_name, question_text, n_results)
        score = []
        for i in range(len(context_list)):
            
            tmp = {
                "question" : question_text,
                "id" : text_data["validation"]["id"][step] + f"_{i}",
                "context" : context_list[i][0]
            }
            score.append(context_list[i][1])
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
    #text_data = run_sparse_retrieval(text_data, training_args) #sparse_retrieval
    #text_data = dpr(text_data) #dense retrieval

    text_data, scores = run_elastic_retrival(text_data) #elasticsearch retrival
    column_names = text_data["validation"].column_names

    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    data_processor = DataProcessor(tokenizer)
    val_text = text_data["validation"]
    val_dataset = data_processor.val_tokenzier(val_text, column_names)
    val_iter = DataLoader(val_dataset, collate_fn = data_collator, batch_size=1)

    return text_data, val_iter, val_dataset, scores

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

def predict(model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device):
    metric = load_metric("squad")
    if "xlm" in model_args.model_name_or_path:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids"])
    else:
        test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])

    model.eval()

    all_start_logits = []
    all_end_logits = []

    t = time.time()
    
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


def make_submission(scores, training_args):
    with open("/opt/ml/lastcode/submission/nbest_predictions.json", "r") as f:
        nbest = json.load(f)

    prediction = dict()
    prev_mrc_id = None
    final_score = []
    score_step = 0

    for mrc_id_step in nbest.keys():
        mrc_id, step = mrc_id_step.split("_")
        if prev_mrc_id != mrc_id:
            if prev_mrc_id is not None:
                large_step = np.argmax(final_score)
                prediction[str(prev_mrc_id)] = nbest[prev_mrc_id+f"_{large_step}"][0]["text"]
                score_step += 1
                final_score = []

            sum_score = sum(scores[score_step])
            prev_mrc_id = mrc_id
        
        # 여기다가 (1 - m) (m)
        final_score.append(nbest[mrc_id_step][0]["probability"] + scores[score_step][int(step)]/sum_score)
    
    with open(training_args.output_dir+f"/final_predictions.json", 'w', encoding='utf-8') as make_file:
        json.dump(prediction, make_file, indent="\t", ensure_ascii=False)
    print(prediction)



def main():
    model_args, data_args, training_args = get_config()
    seed_everything(training_args.seed)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer, model  = get_model(model_args, training_args)
    model.cuda()
    text_data, test_loader, test_dataset, scores = get_data(training_args, tokenizer)
    predict(model, text_data, test_loader, test_dataset, model_args, data_args, training_args, device)

    make_submission(scores, training_args)

if __name__ == "__main__":
    main()