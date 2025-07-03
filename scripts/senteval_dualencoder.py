"""
SentEval pipeline for VisionTextDualEncoder text encoder
"""
import sys
import logging
import torch

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Import your model utils
from src.model.utils import get_model_and_processor

def prepare(params, samples):
    # No preparation needed for transformers
    return

def batcher(params, batch):
    # Join tokens to sentences
    sentences = [' '.join(sent) if sent != [] else '.' for sent in batch]
    # Tokenize and encode
    inputs = params['processor'].tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        # Get text encoder output (assumes model.text_model exists)
        outputs = params['model'].text_model(**inputs)
        # Use [CLS] token embedding as sentence embedding
        hidden_state = outputs.last_hidden_state
        if type(hidden_state) is tuple:
            hidden_state = hidden_state[0]
        embeddings = hidden_state[:, 0, :].cpu().numpy()
    return embeddings

if __name__ == "__main__":
    # Load model and processor
    model, processor = get_model_and_processor(pretrained=True)
    model.eval()
    params_senteval = {
        'task_path': PATH_TO_DATA,
        'usepytorch': True,
        'kfold': 5,
        'model': model,
        'processor': processor
    }
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                     'tenacity': 3, 'epoch_size': 2}

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = [
        'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
        'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
        'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
        'Length', 'WordContent', 'Depth', 'TopConstituents',
        'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
        'OddManOut', 'CoordinationInversion'
    ]
    transfer_tasks = [
        'STS12',
    ]
    results = se.eval(transfer_tasks)
    print(results)
