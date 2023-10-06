from sentence_transformers import LoggingHandler

import logging
from datetime import datetime
import tqdm
import json
import random
from typing import List, Dict, Any
import numpy as np
import pickle

import data_utilities as du
import model_utilities as mu

seed = 777
np.random.seed(seed)
random.seed(seed)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

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

timestr = datetime.now().strftime("%y%m%d_%H%M")
model_save_path = f"{experiments_root}/{model_name}_{dataset_name}_{timestr}"

train_corpus, train_queries, train_triples = du.load_data(datasets_root, dataset_name, p_mask)

train_triples, dev_triples = du.create_train_dev_triples(train_triples)

dev_samples = du.create_dev_data(dev_triples, train_queries, train_corpus, num_dev_queries, num_max_dev_negatives)

train_samples = du.create_train_data(train_corpus, train_triples, dev_samples, train_queries, pos_neg_ration, max_train_samples)

model = mu.get_model(model_name=load_model_name, num_labels=num_labels, max_length=max_length)

train_dataloader = mu.get_train_dataloader(dataset=train_samples, batch_size=train_batch_size)

evaluator = mu.get_evaluator(samples=dev_samples)

# Configure the training
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={'lr': learning_rate},
          use_amp=True)

#Save latest model
# model.save(model_save_path+'-latest')
