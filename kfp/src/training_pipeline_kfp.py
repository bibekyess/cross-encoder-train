from sentence_transformers import LoggingHandler

import logging
from datetime import datetime
import random
from typing import List, Dict, Any, NamedTuple
import torch
import pickle

import data_utilities as du
import model_utilities as mu

from kfp import dsl
from kfp.client import Client
from kfp.dsl import Input, Output, Model, Dataset

import numpy as np
import mlflow
import sentence_transformers
import cloudpickle
from sys import version_info

from kubernetes import client, config
import base64

seed = 777
# np.random.seed(seed)
random.seed(seed)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.WARNING,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


timestr = datetime.now().strftime("%y%m%d_%H%M")

# @dsl.component(base_image='nvidia/cuda:11.8.0-base-ubuntu20.04', target_image='bibekyess/cross-encoder-train:v28')
# def load_data_op(datasets_root: str, dataset_name: str, p_mask: float) -> NamedTuple(
#                 'outputs', train_corpus=Dict[str, str], train_queries=Dict[str, str], train_triples= List[List[str]]):
#     train_corpus, train_queries, train_triples = du.load_data(datasets_root, dataset_name, p_mask)
#     outputs = NamedTuple('outputs', train_corpus=Dict[str, str], train_queries=Dict[str, str], train_triples= List[List[str]])
#     return outputs(train_corpus, train_queries, train_triples)

@dsl.component(base_image='nvidia/cuda:11.8.0-base-ubuntu20.04', target_image='bibekyess/cross-encoder-train:v28')
def load_data_op(datasets_root: str, dataset_name: str, p_mask: float,
                 train_corpus: Output[Dataset], train_queries: Output[Dataset], train_triples: Output[Dataset]) -> None:
    train_corpus_value, train_queries_value, train_triples_value = du.load_data(datasets_root, dataset_name, p_mask)

    with open(train_corpus.path, "wb") as pickle_file:
        pickle.dump(train_corpus_value, pickle_file)

    with open(train_queries.path, "wb") as pickle_file:
        pickle.dump(train_queries_value, pickle_file)    

    with open(train_triples.path, "wb") as pickle_file:
        pickle.dump(train_triples_value, pickle_file)  



@dsl.component(base_image='nvidia/cuda:11.8.0-base-ubuntu20.04', target_image='bibekyess/cross-encoder-train:v28')
def create_train_dev_triples_op(train_triples_all: Input[Dataset], train_triples: Output[Dataset], dev_triples: Output[Dataset]) -> None:
    
    with open(train_triples_all.path, "rb") as pickle_file:
        train_triples_all_value = pickle.load(pickle_file)

    train_triples_value, dev_triples_value = du.create_train_dev_triples(train_triples_all_value)

    with open(train_triples.path, "wb") as pickle_file:
        pickle.dump(train_triples_value, pickle_file)  

    with open(dev_triples.path, "wb") as pickle_file:
        pickle.dump(dev_triples_value, pickle_file)  


@dsl.component(base_image='nvidia/cuda:11.8.0-base-ubuntu20.04', target_image='bibekyess/cross-encoder-train:v28')
def create_dev_data_op(dev_triples: Input[Dataset], 
                       train_queries: Input[Dataset], 
                       train_corpus: Input[Dataset], 
                       num_dev_queries: int, num_max_dev_negatives: int,
                       dev_samples: Output[Dataset]) -> None:
    
    with open(dev_triples.path, "rb") as pickle_file:
        dev_triples_value = pickle.load(pickle_file)

    with open(train_queries.path, "rb") as pickle_file:
        train_queries_value = pickle.load(pickle_file)

    with open(train_corpus.path, "rb") as pickle_file:
        train_corpus_value = pickle.load(pickle_file)
  
    dev_samples_value = du.create_dev_data(dev_triples_value, train_queries_value, train_corpus_value, num_dev_queries, num_max_dev_negatives)

    with open(dev_samples.path, "wb") as pickle_file:
        pickle.dump(dev_samples_value, pickle_file) 


@dsl.component(base_image='nvidia/cuda:11.8.0-base-ubuntu20.04', target_image='bibekyess/cross-encoder-train:v28',
               packages_to_install=['sentence-transformers', 'tqdm'])
def create_train_data_op(dev_samples: Input[Dataset],
                         train_triples: Input[Dataset], 
                         train_queries: Input[Dataset], 
                         train_corpus: Input[Dataset], 
                         pos_neg_ration: int, 
                         max_train_samples: float,
                         train_samples: Output[Dataset])-> None:
    
    with open(dev_samples.path, "rb") as pickle_file:
        dev_samples_value = pickle.load(pickle_file)

    with open(train_triples.path, "rb") as pickle_file:
        train_triples_value = pickle.load(pickle_file)

    with open(train_queries.path, "rb") as pickle_file:
        train_queries_value = pickle.load(pickle_file)

    with open(train_corpus.path, "rb") as pickle_file:
        train_corpus_value = pickle.load(pickle_file)
        
    train_samples_value = du.create_train_data(train_corpus_value, train_triples_value, dev_samples_value, train_queries_value, pos_neg_ration, max_train_samples)

    with open(train_samples.path, "wb") as pickle_file:
        pickle.dump(train_samples_value, pickle_file) 

@dsl.component(base_image='nvidia/cuda:11.8.0-base-ubuntu20.04', target_image='bibekyess/cross-encoder-train:v28',
               packages_to_install=['sentence-transformers'])
def get_model_op(load_model_name: str, num_labels: int, max_length: int, model: Output[Model]) -> None:
    cross_encoder_model = mu.get_model(model_name=load_model_name, num_labels=num_labels, max_length=max_length)
    torch.save(cross_encoder_model, model.path)


# @dsl.component(base_image='nvidia/cuda:11.8.0-base-ubuntu20.04', target_image='bibekyess/cross-encoder-train:v28',
#                packages_to_install=['torch', 'sentence-transformers', 'tqdm', 'numpy'])
# def get_train_dataloader(train_samples: Dict[str, Any], train_batch_size: int):
#     train_dataloader = mu.get_train_dataloader(dataset=train_samples, batch_size=train_batch_size)
#     return train_dataloader


# @dsl.component(base_image='nvidia/cuda:11.8.0-base-ubuntu20.04', target_image='bibekyess/cross-encoder-train:v28',
#                packages_to_install=['torch', 'sentence-transformers', 'tqdm', 'numpy']):
# def get_evaluator(dev_samples):
#     evaluator = mu.get_evaluator(samples=dev_samples)
#     return evaluator

@dsl.component(base_image='nvidia/cuda:11.8.0-base-ubuntu20.04', target_image='bibekyess/cross-encoder-train:v28',
               packages_to_install=['sentence-transformers', 'boto3'],
)
def train_model_op(model: Input[Model], num_epochs: int, evaluation_steps: int, warmup_steps: int,
                learning_rate: float,
                train_samples: Input[Dataset], train_batch_size: int,
                dev_samples: Input[Dataset],
                output_model: Output[Model]) -> None:
    with open(dev_samples.path, "rb") as pickle_file:
        dev_samples_value = pickle.load(pickle_file)

    with open(train_samples.path, "rb") as pickle_file:
        train_samples_value = pickle.load(pickle_file)

    train_dataloader = mu.get_train_dataloader(dataset=train_samples_value, batch_size=train_batch_size)

    evaluator = mu.get_evaluator(samples=dev_samples_value)
    
    cross_encoder_model = torch.load(model.path)

    cross_encoder_model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},
            use_amp=True)
    
    torch.save(cross_encoder_model, output_model.path)


@dsl.component(base_image='nvidia/cuda:11.8.0-base-ubuntu20.04', target_image='bibekyess/cross-encoder-train:v28',
               packages_to_install=['sentence-transformers', 'mlflow', 'cloudpickle', 'boto3'],
)
def log_model_mlflow_registry_op(model: Input[Model]) -> None:

    # mlflow must be running with database backend
    mlflow.set_tracking_uri('http://192.168.0.29:5000') # FIXME
    PYTHON_VERSION = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            f"python={PYTHON_VERSION}",
            "pip",
            {
                "pip": [
                    f"mlflow=={mlflow.__version__}",
                    f"sentence-transformers=={sentence_transformers.__version__}",
                    f"cloudpickle=={cloudpickle.__version__}",
                    f"numpy=={np.__version__}"
                ],
            },
        ],
        "name": "cross_encoder_env",
    }
    cross_encoder_model = torch.load(model.path)
    model_path = './saved_model.pth'
    torch.save(cross_encoder_model, model_path)
    
    # Save the MLflow Model
    mlflow_pyfunc_model_path = "cross_encoder_pyfunc"
    mlflow.pyfunc.log_model(
        artifact_path = mlflow_pyfunc_model_path,
        python_model=mu.CrossEncoderWrapper(),
        artifacts={"model_path": model_path},
        conda_env=conda_env
    )

@dsl.pipeline
def my_pipeline(aws_access_key_id: str, aws_secret_access_key: str) -> None:
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
    create_train_dev_triples_task = create_train_dev_triples_op(train_triples_all=load_data_task.outputs['train_triples'])
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
    log_model_mlflow_registry_task = log_model_mlflow_registry_op(model=train_model_task.output)

    log_model_mlflow_registry_task.set_env_variable(name="MLFLOW_S3_ENDPOINT_URL", value="http://192.168.0.29:9000")
    log_model_mlflow_registry_task.set_env_variable(name="MLFLOW_S3_IGNORE_TLS", value="true")
    log_model_mlflow_registry_task.set_env_variable(name="AWS_ACCESS_KEY_ID", value=aws_access_key_id)
    log_model_mlflow_registry_task.set_env_variable(name="AWS_SECRET_ACCESS_KEY", value=aws_secret_access_key)


    # envs = ["MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID" , "AWS_SECRET_ACCESS_KEY" , "MLFLOW_S3_IGNORE_TLS"]
    # # set_env_variable(name: str, value: str)
    # config.load_kube_config()
    # v1 = client.CoreV1Api()
    # kube_secret = v1.read_namespaced_secret("mlflow-secret", "kubeflow-user-example-com").data
    # for name in envs:
    #     log_model_mlflow_registry_task.set_env_variable(
    #             name=name,
    #             value= base64.b64decode(kube_secret.get(name)).decode('utf-8')
    #             )

def start_training_pipeline_run(aws_access_key_id: str, aws_secret_access_key: str) -> None:
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
    run = client.create_run_from_pipeline_func(my_pipeline, experiment_name=custom_experiment.display_name, enable_caching=False,
                                               arguments={
                                                    'aws_access_key_id': aws_access_key_id, 
                                                    'aws_secret_access_key': aws_secret_access_key
                                               })
    url = f'{HOST}/#/runs/details/{run.run_id}'
    print(url)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws_access_key_id", type=str, required=True, help= "MINIO Access Key ID")
    parser.add_argument("--aws_secret_access_key", type=str, required=True, help= "MINIO Secret Access Key")
    args = parser.parse_args()

    aws_access_key_id = args.aws_access_key_id
    aws_secret_access_key = args.aws_secret_access_key

    start_training_pipeline_run(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)


