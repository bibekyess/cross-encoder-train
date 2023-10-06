from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

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
