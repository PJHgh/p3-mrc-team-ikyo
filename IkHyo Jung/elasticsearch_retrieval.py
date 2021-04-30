import json
import os
import time

from elasticsearch import Elasticsearch
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm

def elastic_setting():
    es_server = Popen(['elasticsearch-7.6.2/bin/elasticsearch'],
                    stdout=PIPE, stderr=STDOUT,
                    preexec_fn=lambda: os.setuid(1)  # as daemon
                    )
    # wait until ES has started
    time.sleep(30)

    es, index_name = elastic_index()
    wiki_path="/opt/ml/input/data/data/wikipedia_documents.json"
    add_wiki_elastic(es, index_name, wiki_path)
    
    return es, index_name

def elastic_index():
    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])

    index_config = {
    "settings": {
        "analysis": {
            "analyzer": {
                "standard_analyzer": {
                    "type": "standard"
                }
            }
        }
    },
    "mappings": {
        "dynamic": "strict", 
        "properties": {
            "document_text": {"type": "text", "analyzer": "standard_analyzer"}
            }
        }
    }
    index_name = 'squad-standard-index'
    es.indices.create(index=index_name, body=index_config, ignore=400)

    return es, index_name

def populate_index(es_obj, index_name, evidence_corpus):

    for i, rec in enumerate(tqdm(evidence_corpus)):
    
        try:
            index_status = es_obj.index(index=index_name, id=i, body=rec)
        except:
            print(f'Unable to load document {i}.')
            
    n_records = es_obj.count(index=index_name)['count']
    print(f'Succesfully loaded {n_records} into {index_name}')


def search_es(es_obj, index_name, question_text, n_results):
    query = {
            'query': {
                'match': {
                    'document_text': question_text
                    }
                }
            }
    
    res = es_obj.search(index=index_name, body=query, size=n_results)
    
    return res

def add_wiki_elastic(es, index_name, wiki_path):
    with open(wiki_path, "r") as f:
        wiki = json.load(f)
    wiki_contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))
    wiki_articles = [{"document_text" : wiki_contexts[i]} for i in range(len(wiki_contexts))]
    populate_index(es, index_name, wiki_articles)

def elastic_retrieval(es, index_name, question_text, n_results):
    res = search_es(es, index_name, question_text, n_results)
    context_list = list((hit['_source']['document_text'], hit['_score']) for hit in res['hits']['hits'])
    return context_list

if __name__ == "__main__":
    es, index_name = elastic_setting()
    question_text = "대한민국의 대통령은?"
    context_list = elastic_retrieval(es, index_name, question_text, n_result)
    print(context_list)