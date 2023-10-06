from sentence_transformers import LoggingHandler

import logging
from datetime import datetime
import random
from typing import List, Dict, Any, NamedTuple
import torch


import data_utilities as du
import model_utilities as mu

from kfp import dsl
from kfp.client import Client
from kfp.dsl import Input, Output, Model

seed = 777
# np.random.seed(seed)
random.seed(seed)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


timestr = datetime.now().strftime("%y%m%d_%H%M")

@dsl.component(base_image='python:3.11-slim', target_image='bibekyess/cross-encoder-train:v3')
def load_data_op(datasets_root: str, dataset_name: str, p_mask: float) -> NamedTuple(
                'outputs', train_corpus=Dict[str, str], train_queries=Dict[str, str], train_triples= List[List[str]]):
    train_corpus, train_queries, train_triples = du.load_data(datasets_root, dataset_name, p_mask)
    outputs = NamedTuple('outputs', train_corpus=Dict[str, str], train_queries=Dict[str, str], train_triples= List[List[str]])
    return outputs(train_corpus, train_queries, train_triples)


@dsl.component(base_image='python:3.11-slim', target_image='bibekyess/cross-encoder-train:v3')
def create_train_dev_triples_op(train_triples: List[List[str]]) -> NamedTuple( 
                                'outputs', train_triples= List[List[str]], dev_triples= List[List[str]]
                                ):
    train_triples, dev_triples = du.create_train_dev_triples(train_triples)
    outputs = NamedTuple( 'outputs', train_triples= List[List[str]], dev_triples= List[List[str]])
    return outputs(train_triples, dev_triples)


@dsl.component(base_image='python:3.11-slim', target_image='bibekyess/cross-encoder-train:v3')
def create_dev_data_op(dev_triples: List[List[str]], 
                       train_queries: Dict[str, str], 
                       train_corpus: Dict[str, str], 
                       num_dev_queries: int, num_max_dev_negatives: int) -> Dict[str, Any]:
    dev_samples = du.create_dev_data(dev_triples, train_queries, train_corpus, num_dev_queries, num_max_dev_negatives)
    return dev_samples

@dsl.component(base_image='python:3.11-slim', target_image='bibekyess/cross-encoder-train:v3',
               packages_to_install=['sentence-transformers==2.2.2', 'tqdm==4.66.1'])
def create_train_data_op(dev_samples: Dict[str, Any],
                         train_triples: List[List[str]], 
                         train_queries: Dict[str, str], 
                         train_corpus: Dict[str, str], 
                         pos_neg_ration: int, 
                         max_train_samples: float)-> Dict[str, Any]:
    train_samples = du.create_train_data(train_corpus, train_triples, dev_samples, train_queries, pos_neg_ration, max_train_samples)
    return train_samples

@dsl.component(base_image='python:3.11-slim', target_image='bibekyess/cross-encoder-train:v3',
               packages_to_install=['sentence-transformers==2.2.2'])
def get_model_op(load_model_name: str, num_labels: int, max_length: int, model: Output[Model]) -> None:
    cross_encoder_model = mu.get_model(model_name=load_model_name, num_labels=num_labels, max_length=max_length)
    torch.save(cross_encoder_model, model.path)


# @dsl.component(base_image='python:3.11-slim', target_image='bibekyess/cross-encoder-train:v3',
#                packages_to_install=['torch', 'sentence-transformers', 'tqdm', 'numpy'])
# def get_train_dataloader(train_samples: Dict[str, Any], train_batch_size: int):
#     train_dataloader = mu.get_train_dataloader(dataset=train_samples, batch_size=train_batch_size)
#     return train_dataloader


# @dsl.component(base_image='python:3.11-slim', target_image='bibekyess/cross-encoder-train:v3',
#                packages_to_install=['torch', 'sentence-transformers', 'tqdm', 'numpy']):
# def get_evaluator(dev_samples):
#     evaluator = mu.get_evaluator(samples=dev_samples)
#     return evaluator

@dsl.component(base_image='python:3.11-slim', target_image='bibekyess/cross-encoder-train:v3',
               packages_to_install=['torch==2.1.0', 'sentence-transformers==2.2.2'])
def train_model_op(model: Input[Model], num_epochs: int, evaluation_steps: int, warmup_steps: int,
                learning_rate: float,
                train_samples: Dict[str, Any], train_batch_size: int,
                dev_samples: Dict[str, Any]) -> None:
    train_dataloader = mu.get_train_dataloader(dataset=train_samples, batch_size=train_batch_size)

    evaluator = mu.get_evaluator(samples=dev_samples)
    
    cross_encoder_model = torch.load(model.path)
    # Train the model
    cross_encoder_model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},
            use_amp=True)


@dsl.pipeline
def my_pipeline() -> None:
    #First, we define the transformer model we want to fine-tune
    model_name = 'distill-kobert'
    load_model_name = "monologg/distilkobert"
    dataset_name = 'ISU_retrieval_data'
    experiments_root = './_saved_models'
    datasets_root = './_processed_datasets'

    # model settings
    max_length = 512
    num_labels = 1

    # training settings
    p_mask = 0.1
    train_batch_size = 32
    num_epochs = 10
    evaluation_steps = 1000
    warmup_steps = 1000
    learning_rate = 2e-05

    # data settings
    pos_neg_ration = 4
    max_train_samples = 1e7
    num_dev_queries = 2000
    num_max_dev_negatives = 20

    load_data_task = load_data_op(datasets_root=datasets_root, dataset_name=dataset_name, p_mask=p_mask)
    create_train_dev_triples_task = create_train_dev_triples_op(train_triples=load_data_task.outputs['train_triples'])
    create_dev_data_task = create_dev_data_op(dev_triples=create_train_dev_triples_task.outputs['dev_triples'], 
                                            train_queries=load_data_task.outputs['train_queries'], 
                                            train_corpus= load_data_task.outputs['train_corpus'], 
                                            num_dev_queries = num_dev_queries, num_max_dev_negatives=num_max_dev_negatives)
    create_train_data_task = create_train_data_op(dev_samples=create_dev_data_task.output,
                                                train_triples= create_train_dev_triples_task.outputs['train_triples'], 
                                                train_queries=load_data_task.outputs['train_queries'], 
                                                train_corpus= load_data_task.outputs['train_corpus'], 
                                                pos_neg_ration= pos_neg_ration, 
                                                max_train_samples = max_train_samples)
    get_model_task = get_model_op(load_model_name=load_model_name, num_labels=num_labels, max_length=max_length)
    train_model_task = train_model_op(model=get_model_task.output, 
                                    num_epochs=num_epochs, 
                                    evaluation_steps=evaluation_steps, 
                                    warmup_steps=warmup_steps,
                                    learning_rate=learning_rate,
                                    train_samples= create_train_data_task.output, 
                                    train_batch_size=train_batch_size,
                                    dev_samples= create_dev_data_task.output
                                    )

def start_training_pipeline_run():
    import requests

    USERNAME = "user@example.com"
    PASSWORD = "12341234" 
    NAMESPACE = "kubeflow-user-example-com"
    HOST = "http://192.168.0.33:8084" # your istio-ingressgateway pod ip:8080

    session = requests.Session()
    response = session.get(HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]

    client = Client(
        host=f"{HOST}/pipeline",
        namespace=f"{NAMESPACE}",
        cookies=f"authservice_session={session_cookie}",
    )

    custom_experiment = client.create_experiment('Custom Registry tracker', namespace=NAMESPACE)

    print(client.list_experiments())
    run = client.create_run_from_pipeline_func(my_pipeline, experiment_name=custom_experiment.display_name, enable_caching=False)
    url = f'{HOST}/#/runs/details/{run.run_id}'
    print(url)

if __name__ == '__main__':
    start_training_pipeline_run()


