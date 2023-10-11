from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator


from typing import Dict, Any, Optional
import numpy as np
import mlflow
import pandas as pd
import torch

#We set num_labels=1, which predicts a continous score between 0 and 1
def get_model(model_name, num_labels, max_length):
    return CrossEncoder(model_name=model_name, num_labels=num_labels, max_length=max_length)

# We create a DataLoader to load our train samples
def get_train_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
def get_evaluator(samples, name='train-eval', mrr_at_k=10):
    return CERerankingEvaluator(samples=samples, name=name, mrr_at_k=mrr_at_k)

class CrossEncoderWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Dict[str, Any]) -> bool:
        # Load SentenceTransformer from disk.
        self.model = torch.load(context.artifacts["model_path"])
        self.ready = True
        return self.ready
    
    def predict(self, context: Dict[str, Any], model_input: pd.DataFrame,  params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        
        print(model_input)

        inputs = model_input.get("inputs")
        # InferenceRequest is not automatically decoded
        if inputs:
            model_input_ = {}
            for inp in inputs:
                model_input_[inp['name']] = inp['data']
            model_input = model_input_

        query = model_input["query"]
        paragraphs = model_input["paragraphs"]
        input = []
        for p in paragraphs:
            input.append([query[0], p])  
        scores = self.model.predict(input)
        return scores