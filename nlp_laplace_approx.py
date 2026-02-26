import os
import io
import json
import random
import argparse
import subprocess
import sys
import gc
import pickle
import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torchnlp.datasets import snli_dataset
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import function_nlp as pf
from datasets import load_dataset
from torch.utils.data import DataLoader
# from laplace import Laplace
from laplace import Laplace
from laplace.curvature import AsdlGGN
from laplace.curvature.backpack import BackPackGGN

from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorWithPadding


class MyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.hf_model = small_model

    def forward(self, data) -> torch.Tensor:
        device = next(self.parameters()).device
        self.data = data
        input_ids = self.data['input_ids'].to(device)
        token_type_ids = self.data['token_type_ids'].to(device)
        attention_mask = self.data['attention_mask'].to(device)
        logits = self.hf_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
        return logits



def parseArgs():

    parser = argparse.ArgumentParser(
        description="Training for calibration distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='cifar_10', help='dataset for training')
    parser.add_argument("--model", type=str, default='resnet_18', help='network architecture for training')
    parser.add_argument("--random_seed", type=int, default=1, help='random seed for training')
    parser.add_argument("--epoch", type=int, default=350, help='epoch for training')
    parser.add_argument("--test_index", type=int, default=100, help='data index for plotting')
    parser.add_argument("--alpha_div", type=float, default=1.2, help='alpha for alpha_divergence')
    parser.add_argument("--alpha_quant", type=float, default=0.1, help='alpha quantile for NC score')
    parser.add_argument("--batch_size", type=int, default=1000, help='batch size for splitting test data')
    parser.add_argument("--temp", type=float, default=1, help='temperature for softmax layer')
    parser.add_argument("--calibration_size", type=int, default=500, help='CP calibration data size')
    parser.add_argument("--large_model", type=str, default='cross-encoder/nli-deberta-v3-large', help='pre-trained large model')
    parser.add_argument("--small_model", type=str, default='cross-encoder/nli-deberta-v3-small', help='pre-trained small model')
    parser.add_argument('--conf', type=float, default=0.9995, help='confidence level for clip the parameters')
    parser.add_argument('--bitwidth', type=int, default=20, help='fine tune the quantized parameters to avoid some strange happens')

    return parser.parse_args()



def split_into_batches(encodings, batch_size):
    # Create batches from encoded data
    if 'token_type_ids' in encodings:
        input_ids_batches = encodings['input_ids'].split(batch_size)
        token_type_ids_batches = encodings['token_type_ids'].split(batch_size)
        attention_mask_batches = encodings['attention_mask'].split(batch_size)

        # Return list of batches
        return [{'input_ids': batch, 'token_type_ids': token_type, 'attention_mask': mask} for batch, token_type, mask in
                zip(input_ids_batches, token_type_ids_batches, attention_mask_batches)]
    else:
        input_ids_batches = encodings['input_ids'].split(batch_size)
        attention_mask_batches = encodings['attention_mask'].split(batch_size)

        # Return list of batches
        return [{'input_ids': batch, 'attention_mask': mask} for batch, mask
                in
                zip(input_ids_batches, attention_mask_batches)]

def quantile(input, level):
    if len(input.shape) == 2:
        num_param = input.shape[0] * input.shape[1]
    elif len(input.shape) == 1:
        num_param = input.shape[0]
    else:
        raise NotImplementedError
    sorted, indices = torch.sort(torch.flatten(input))

    lo = int(np.floor((1 - level) * num_param))
    hi = int(np.ceil(level * num_param)) - 1

    return sorted[lo], sorted[hi]


def unif_quan(input, bitwidth, conf):

    m, M = quantile(input, conf)
    trimmed_input = torch.clip(input, m, M)
    resol = (M - m) / ((2 ** bitwidth) - 1)
    return torch.round((trimmed_input - m) / resol) * resol + m

def model_quantization(model, conf=None, bitwidth=None):
    if bitwidth is None:
        model = model
    else:
        for f in model.parameters():
            f.data = unif_quan(f.data, bitwidth, conf)
    return model


def main(args):

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    torch.manual_seed(10)

    '''
            Load the data: SNLI
    '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Current deploy device is {device}')

    # Download the SNLI data set at the first time
    data = snli_dataset(directory='data/',
                     train=True,
                     dev=True,
                     test=True,
                     train_filename='snli_1.0_train.jsonl',
                     dev_filename='snli_1.0_dev.jsonl',
                     test_filename='snli_1.0_test.jsonl',
                     extracted_name='snli_1.0',
                     check_files=['snli_1.0/snli_1.0_train.jsonl'],
                     url='http://nlp.stanford.edu/projects/snli/snli_1.0.zip'
                        )


    # Load the tokenizer and model: using nli-deberta_v3_large as the large model

    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model, force_download=False)
    small_model = AutoModelForSequenceClassification.from_pretrained(args.small_model, force_download=False)
    small_model = model_quantization(small_model, conf=args.conf, bitwidth=args.bitwidth)

    # Load the model to CUDA device: Multiply GPUs or just single
    if torch.cuda.device_count() > 1:
        small_model = torch.nn.DataParallel(small_model)
        print(f'Parallel inference on {torch.cuda.device_count()} GPUs')

    total_params_small = sum(p.numel() for p in small_model.parameters() if p.requires_grad)
    print(f'Number of parameter for small model {total_params_small}')


    '''
            Combine and load the data set, preparing for tokenizer and model
    '''

    training_data_size, dev_data_size, test_data_size = len(data[0]), len(data[1]), len(data[2])
    print(f'Training data size {training_data_size}, validation size {dev_data_size}, test data size {test_data_size}')

    data_set_key = ['premise', 'hypothesis', 'label']
    training_data_set = {k: [] for k in data_set_key}
    dev_data_set = {k: [] for k in data_set_key}
    test_data_set = {k: [] for k in data_set_key}

    train_data = [example for example in data[0] if example['label'] != '-']
    dev_data = [example for example in data[1] if example['label'] != '-']
    test_data = [example for example in data[2] if example['label'] != '-']

    training_data_size, dev_data_size, test_data_size = len(train_data), len(dev_data), len(test_data)
    print(f'Training data size {training_data_size}, validation size {dev_data_size}, test data size {test_data_size}')

    # Randomly sample 500 data from Dev as the calibration data set
    calibration_data_size = args.calibration_size


    for name in data_set_key:
        for i in range(training_data_size):
            training_data_set[name].append(train_data[i][name])
        for i in range(calibration_data_size):
            dev_data_set[name].append(dev_data[i][name])
        for i in range(test_data_size):
            test_data_set[name].append(test_data[i][name])


    # small_training_input = small_tokenizer(training_data_set['premise'], training_data_set['hypothesis'], padding=True, truncation=True, return_tensors="pt")
    small_dev_input = small_tokenizer(dev_data_set['premise'], dev_data_set['hypothesis'], padding=True, truncation=True, return_tensors="pt")
    small_test_input = small_tokenizer(test_data_set['premise'], test_data_set['hypothesis'], padding=True, truncation=True, return_tensors="pt")

    small_test_input = split_into_batches(small_test_input, batch_size=100)

    label_mapping = ['contradiction', 'entailment', 'neutral']
    label_to_id = {label: idx for idx, label in enumerate(label_mapping)}
    label_ids = [label_to_id.get(lab, -1) for lab in dev_data_set['label']]
    labels = torch.tensor(label_ids)
    print(labels.shape)


    label_ids = [label_to_id.get(lab, -1) for lab in test_data_set['label']]
    test_labels = torch.tensor(label_ids).cuda()


    print(f'Evaluate the small model for the test data set')

    small_model.eval()
    small_model.to(device)
    small_test_confs, small_test_pred_label = torch.tensor([]).to(device), []
    small_test_probs, small_test_embeddings = torch.tensor([]).to(device), torch.tensor([]).to(device)
    for current_batch in small_test_input:
        current_batch['input_ids'] = current_batch['input_ids'].to(device)
        if 'token_type_ids' in current_batch:
            current_batch['token_type_ids'] = current_batch['token_type_ids'].to(device)
        current_batch['attention_mask'] = current_batch['attention_mask'].to(device)

        small_test_logits = small_model(**current_batch).logits
        small_test_prob = torch.nn.functional.softmax(small_test_logits.data, dim=1)

        conf, label = torch.max(small_test_prob, dim=1)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        label = [label_mapping[score_max] for score_max in label]

        small_test_confs = torch.cat((small_test_confs, conf))
        small_test_pred_label.append(label)
        small_test_probs = torch.cat((small_test_probs, small_test_prob))

    ECE, ACC, _ = pf.expected_calibration_error(small_test_confs, small_test_pred_label, test_data_set['label'],
                                                batch=True, num_bins=15)
    print(f'current small test ECE is {ECE * 100}%, acc is {ACC * 100}%')





    ''''
        Test the Laplace approximation
    '''
    dataset_dict = {
        "input_ids": small_dev_input['input_ids'].tolist(),
        "token_type_ids": small_dev_input['token_type_ids'].tolist(),
        "attention_mask": small_dev_input['attention_mask'].tolist(),
        "labels": labels.tolist()

    }

    dev_dataset = Dataset.from_dict(dataset_dict)
    dev_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=100, shuffle=False)
    data = next(iter(dev_loader))

    model = MyModel()
    model.eval()

    print(f'fitting the Laplace approx.')
    la = Laplace(
        model,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="full",
        feature_reduction="pick_last",
    )
    la.fit(dev_loader)
    la.optimize_prior_precision()

    print(f'Laplace fitting done')


    small_test_input = small_tokenizer(test_data_set['premise'], test_data_set['hypothesis'], padding=True, truncation=True, return_tensors="pt")

    dataset_dict = {
        "input_ids": small_test_input['input_ids'].tolist(),
        "token_type_ids": small_test_input['token_type_ids'].tolist(),
        "attention_mask": small_test_input['attention_mask'].tolist(),
        "labels": test_labels.tolist()

    }

    test_dataset = Dataset.from_dict(dataset_dict)
    test_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    results = torch.tensor([]).cuda()

    for data in test_loader:
        pred_la = la(data, pred_type="glm", link_approx="probit")
        results = torch.cat((results, pred_la))

    results = results.reshape(-1, 3)

    conf, label = torch.max(results, dim=1)
    ECE, ACC, _ = pf.expected_calibration_error(conf, label, test_dataset['labels'], batch=False, num_bins=15)
    print(f'ECE is {100 * ECE}, Acc is {100 * ACC}')





if __name__ == '__main__':
    args = parseArgs()
    main(args)