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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torchnlp.datasets import snli_dataset
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import function_nlp as pf
from datasets import load_dataset


def parseArgs():

    parser = argparse.ArgumentParser(
        description="Training for calibration distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='cifar_10', help='dataset for training')
    parser.add_argument("--model", type=str, default='resnet_18', help='network architecture for training')
    parser.add_argument("--random_seed", type=int, default=1, help='random seed for training')
    parser.add_argument("--epoch", type=int, default=350, help='epoch for training')
    parser.add_argument("--first_milestone", type=int, default=150, help='first learning rate change')
    parser.add_argument("--second_milestone", type=int, default=250, help='second learning rate change')
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate for training')
    parser.add_argument("--momentum", type=float, default=0.9, help='momentum for training')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help='weight decay for training')
    parser.add_argument("--optimizer", type=str, default='sgd', help='optimizer for training')
    parser.add_argument("--test_index", type=int, default=100, help='data index for plotting')
    parser.add_argument("--alpha_div", type=float, default=1.5, help='alpha for alpha_divergence')
    parser.add_argument("--alpha_quant", type=float, default=0.1, help='alpha quantile for NC score')
    parser.add_argument("--batch_size", type=int, default=1000, help='batch size for splitting test data')
    parser.add_argument("--temp", type=float, default=1, help='temperature for softmax layer')
    parser.add_argument("--calibration_size", type=int, default=1000, help='CP calibration data size')
    parser.add_argument("--large_model", type=str, default='cross-encoder/nli-deberta-v3-large', help='pre-trained large model')
    parser.add_argument("--small_model", type=str, default='cross-encoder/nli-deberta-v3-small', help='pre-trained small model')
    parser.add_argument('--quantize', type=str, default=torch.quint8, help='weight quantized')
    parser.add_argument('--conf', type=float, default=0.95, help='confidence level for clip the parameters')
    parser.add_argument('--bitwidth', type=float, default=2, help='fine tune the quantized parameters to avoid some strange happens')

    return parser.parse_args()

def D_alpha(p, q, alpha):
    """alpha divergence of two discrete distribution
    """

    new_p, new_q = np.power(p, alpha), np.power(q, 1 - alpha)
    distance_p_q = np.log2(np.sum(np.multiply(new_p, new_q))) / (alpha - 1)

    return distance_p_q

def compute_quantile(scores, alpha):
    """compute quantile from the scores
    """
    n = len(scores)
    return np.quantile(scores, np.ceil((n+1)*(1-alpha))/n, method="inverted_cdf")

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


def plot_nli_distribution(dist, prediction, true, cvg, efficiency, args, labels=['Entailment', 'Neutral', 'Contradiction']):

    # Define the triangle's vertices
    vertices = np.array([
        [0, 0],  # Contradiction
        [1, 0],  # Entailment
        [0.5, np.sqrt(3) / 2]  # Neutral
    ])

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Draw the triangle
    triangle = plt.Polygon(vertices, edgecolor='black', fill=None)
    ax.add_patch(triangle)

    # Label the vertices
    margin = 0.1
    ax.text(vertices[0][0] - margin, vertices[0][1] - margin, labels[0], horizontalalignment='right')
    ax.text(vertices[1][0] + margin, vertices[1][1] - margin, labels[1], horizontalalignment='left')
    ax.text(vertices[2][0], vertices[2][1] + margin, labels[2], horizontalalignment='center', verticalalignment='bottom')

    # Calculate the point in the triangle
    point = prediction[0] * vertices[0] + prediction[1] * vertices[1] + prediction[2] * vertices[2]
    print(point)
    # Plot the point
    ax.plot(point[0], point[1], 'o', color='blue', alpha=1, markersize=3)

    point = true[0] * vertices[0] + true[1] * vertices[1] + true[2] * vertices[2]
    print(point)
    # Plot the point
    ax.plot(point[0], point[1], 's', color='red', alpha=1, markersize=3)

    for index in range(dist.shape[0]):
        point = dist[index][0] * vertices[0] + dist[index][1] * vertices[1] + dist[index][2] * vertices[2]
        # print(point)
        # Plot the point
        ax.plot(point[0], point[1], 'o', color='green', alpha=0.3, markersize=0.4)

    plt.text(0.3, -0.1, 'Trained by Transformer')
    plt.text(0.3, -0.2, 'Score function is KL')

    # plt.text(0.8,0.6, 'Test data index:{}'.format(args.test_index))
    plt.text(0.8,0.5, 'Per-data efficiency:{}%'.format(round(efficiency*100, 2)))
    plt.text(0.8,0.4, 'Coverage:{}%'.format(round(cvg*100, 2)))
    #
    # # Plot the point
    # ax.plot(point[0], point[1], 'o', color='green')

    # Optional: annotate the point with the distribution

    # ax.annotate(f'{dist}', xy=(point[0], point[1]), xytext=(point[0] + 0.05, point[1]),
    #             textcoords='offset points', ha='left', va='bottom')

    # Set limits and aspect
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')  # Hide the axes
    plt.savefig('./nlp_test_data_{}.jpeg'.format(args.test_index), dpi=1000)

    plt.show()


import torch
import numpy as np
from scipy.stats import norm


def logquan(input, bitwidth, M):
    assert bitwidth >= 2
    m = M - 2 ** (bitwidth - 2) + 1
    quan_abs = torch.min(torch.round(torch.log2(torch.abs(input))), M)
    power_of_two = m + torch.relu(quan_abs - m) - torch.nan_to_num(torch.relu(m - quan_abs) * float("Inf"), nan=0.0)
    return torch.sign(input) * (2 ** power_of_two)

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
    torch.manual_seed(1)

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
                     url='http://nlp.stanford.edu/projects/snli/snli_1.0.zip')


    # Load the tokenizer and model: using nli-deberta_v3_large as the large model
    large_tokenizer = AutoTokenizer.from_pretrained(args.large_model, force_download=False)
    large_model = AutoModelForSequenceClassification.from_pretrained(args.large_model, force_download=False)

    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model, force_download=False)
    small_model = AutoModelForSequenceClassification.from_pretrained(args.small_model, force_download=False)

    # Load the model to CUDA device: Multiply GPUs or just single
    if torch.cuda.device_count() > 1:
        large_model = torch.nn.DataParallel(large_model)
        small_model = torch.nn.DataParallel(small_model)
        print(f'Parallel inference on {torch.cuda.device_count()} GPUs')

    total_params_large = sum(p.numel() for p in large_model.parameters() if p.requires_grad)
    total_params_small = sum(p.numel() for p in small_model.parameters() if p.requires_grad)
    print(f'Number of parameter for large model {total_params_large}')
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

    # Randomly sample 500 data from Dev as the calibration data set
    calibration_data_size = args.calibration_size
    dev_data_sample_indices = random.sample(range(dev_data_size), calibration_data_size)

    for name in data_set_key:
        for i in range(training_data_size):
            training_data_set[name].append(data[0][i][name])
        for i in range(calibration_data_size):
            dev_data_set[name].append(data[1][dev_data_sample_indices[i]][name])
        for i in range(test_data_size):
            test_data_set[name].append(data[2][i][name])

    large_dev_input = large_tokenizer(dev_data_set['premise'], dev_data_set['hypothesis'], padding=True, truncation=True, return_tensors="pt")
    large_test_input = large_tokenizer(test_data_set['premise'], test_data_set['hypothesis'], padding=True, truncation=True, return_tensors="pt")
    large_test_input = split_into_batches(large_test_input, batch_size=args.batch_size)

    small_dev_input = small_tokenizer(dev_data_set['premise'], dev_data_set['hypothesis'], padding=True, truncation=True, return_tensors="pt")
    small_test_input = small_tokenizer(test_data_set['premise'], test_data_set['hypothesis'], padding=True, truncation=True, return_tensors="pt")
    small_test_input = split_into_batches(small_test_input, batch_size=args.batch_size)

    '''
            Evaluate the small model for the calibration data set
    '''
    small_dev_input = small_dev_input.to(device)
    small_model = model_quantization(small_model, conf=args.conf, bitwidth=args.bitwidth)
    small_model.to(device)
    small_model.eval()
    pred_confs, pred_labels = torch.tensor([]).to(device), []
    with torch.no_grad():
        small_calibration_logits = small_model(**small_dev_input).logits / args.temp

        conf, label = torch.max(torch.nn.functional.softmax(small_calibration_logits.data, dim=1).data, dim=1)
        small_calibration_probs = torch.nn.functional.softmax(small_calibration_logits.data, dim=1).detach().cpu().numpy()
        label_mapping = ['contradiction', 'entailment', 'neutral']
        small_calibration_labels = [label_mapping[score_max] for score_max in label]

        pred_confs = torch.cat((pred_confs, conf))
        pred_labels.append(small_calibration_labels)

    ECE, ACC, _ = pf.expected_calibration_error(pred_confs, pred_labels, dev_data_set['label'], batch=True,  num_bins=15)
    print(f'small model ECE is {ECE * 100}%, with accuracy as {ACC*100}%')

    del small_dev_input, small_calibration_logits, small_model
    torch.cuda.empty_cache()
    gc.collect()

    '''
            Evaluate the large model for the calibration data set
    '''
    device = 'cuda'
    large_dev_input = large_dev_input.to(device)
    large_model = model_quantization(large_model, conf=1, bitwidth=32)
    large_model.to(device)
    large_model.eval()
    pred_confs, pred_labels = torch.tensor([]).to(device), []
    with torch.no_grad():
        large_calibration_logits = large_model(**large_dev_input).logits
        large_calibration_probs = torch.nn.functional.softmax(large_calibration_logits.data, dim=1).detach().cpu().numpy()
        large_calibration_pred_conf, large_calibration_pred_label = torch.max(torch.nn.functional.softmax(large_calibration_logits.data, dim=1).data, dim=1)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        large_calibration_labels = [label_mapping[score_max] for score_max in large_calibration_pred_label]

        pred_confs = torch.cat((pred_confs, large_calibration_pred_conf))
        pred_labels.append(large_calibration_labels)

        ECE, ACC, _ = pf.expected_calibration_error(pred_confs, pred_labels, dev_data_set['label'], batch=True, num_bins=15)
        print(f'Before TS: large model ECE is {ECE*100}%, ACC is {ACC*100}%')

        '''
            Do the TS to calibrate the large-scale model
        '''
        ece_val = 10 ** 7
        T_opt_ece = 1.0
        T = 0.1

        # for i in range(40): # rang(40)
        #     temperature = T
        #     TS_logits = large_model(**large_dev_input).logits / temperature
        #     TS_pred_conf, TS_pred_label = torch.max(torch.nn.functional.softmax(TS_logits.data, dim=1).data, dim=1)
        #     label_mapping = ['contradiction', 'entailment', 'neutral']
        #     TS_labels = [label_mapping[score_max] for score_max in TS_pred_label]
        #
        #     ECE, ACC, _ = pf.expected_calibration_error(TS_pred_conf, TS_labels, dev_data_set['label'],  batch=False, num_bins=15)
        #     print(f'Current TS with TEMP {temperature}: large model ECE is {ECE * 100}%, ACC is {ACC*100}%')
        #
        #     if ece_val > ECE:
        #         T_opt_ece = T
        #         ece_val = ECE
        #     T += 0.1

        T_opt_ece = 2.1
        large_calibration_logits = large_model(**large_dev_input).logits / T_opt_ece
        large_calibration_probs = torch.nn.functional.softmax(large_calibration_logits.data, dim=1).detach().cpu().numpy()

        print(f'The best Temperature for TS is {T_opt_ece}')



    del large_calibration_logits, large_dev_input, large_model
    torch.cuda.empty_cache()
    gc.collect()

    '''
            Compute the alpha and KL divergence based on the calibration data set
    '''

    # distance_function = ['alpha', 'inner', 'tv', 'kl', 'ws']
    distance_function = ['kl']
    calibration_score = {k: [] for k in distance_function}
    calibration_score_quantile = {k: [] for k in distance_function}
    for i in range(large_calibration_probs.shape[0]):
        calibration_score['kl'].append(entropy(large_calibration_probs[i], small_calibration_probs[i], base=2))

    # Calculate the alpha quantile based on the calibration NC score
    for name in distance_function:
        calibration_score_quantile[name] = compute_quantile(calibration_score[name], args.alpha_quant)
        print(f'NC function: {name}, alpha quantile: {calibration_score_quantile[name]}')

    del large_calibration_probs, small_calibration_probs
    torch.cuda.empty_cache()
    gc.collect()

    '''
              Evaluate the large model for the test data set
    '''
    print(f'Evaluate the large model for the test data set')
    large_model = AutoModelForSequenceClassification.from_pretrained(args.large_model, force_download=False)
    large_model = model_quantization(large_model, conf=1, bitwidth=32)

    large_model.to(device)
    large_model.eval()

    with torch.no_grad():
        large_test_confs, large_test_pred_label = torch.tensor([]).to(device), []
        large_test_probs = []
        for current_batch in large_test_input:
            current_batch['input_ids'] = current_batch['input_ids'].to(device)
            if 'token_type_ids' in current_batch:
                current_batch['token_type_ids'] = current_batch['token_type_ids'].to(device)
            current_batch['attention_mask'] = current_batch['attention_mask'].to(device)

            large_test_logits = large_model(**current_batch).logits / T_opt_ece
            large_test_prob = torch.nn.functional.softmax(large_test_logits.data, dim=1).detach().cpu().numpy().tolist()

            conf, label = torch.max(torch.nn.functional.softmax(large_test_logits.data, dim=1).data, dim=1)
            label_mapping = ['contradiction', 'entailment', 'neutral']
            label = [label_mapping[score_max] for score_max in label]

            large_test_confs = torch.cat((large_test_confs, conf))
            large_test_pred_label.append(label)
            large_test_probs.append(large_test_prob)

        ECE, ACC, _ = pf.expected_calibration_error(large_test_confs, large_test_pred_label, test_data_set['label'], batch=True, num_bins=15)
        print(f'current large model with TS: ECE is {ECE*100}%, acc is {ACC*100}%')

        # pf.reliability_diagram_plot(large_test_confs, large_test_pred_label, test_data_set['label'], network_type='large', batch=True, num_bins=15)

    '''
            Evaluate the small model for the test data set
    '''
    print(f'Evaluate the small model for the test data set')
    small_model = AutoModelForSequenceClassification.from_pretrained(args.small_model, force_download=False)
    small_model = model_quantization(small_model, conf=args.conf, bitwidth=args.bitwidth)
    small_model.to(device)
    small_model.eval()

    with torch.no_grad():
        small_test_confs, small_test_pred_label = torch.tensor([]).to(device), []
        small_test_probs = []
        for current_batch in small_test_input:
            current_batch['input_ids'] = current_batch['input_ids'].to(device)
            if 'token_type_ids' in current_batch:
                current_batch['token_type_ids'] = current_batch['token_type_ids'].to(device)
            current_batch['attention_mask'] = current_batch['attention_mask'].to(device)

            small_test_logits = small_model(**current_batch).logits / args.temp
            small_test_prob = torch.nn.functional.softmax(small_test_logits.data, dim=1).detach().cpu().numpy().tolist()

            conf, label = torch.max(torch.nn.functional.softmax(small_test_logits.data, dim=1).data, dim=1)
            label_mapping = ['contradiction', 'entailment', 'neutral']
            label = [label_mapping[score_max] for score_max in label]

            small_test_confs = torch.cat((small_test_confs, conf))
            small_test_pred_label.append(label)
            small_test_probs.append(small_test_prob)

        ECE, ACC, _ = pf.expected_calibration_error(small_test_confs, small_test_pred_label, test_data_set['label'], batch=True, num_bins=15)
        print(f'current small ECE is {ECE*100}%, acc is {ACC*100}%')

        # pf.reliability_diagram_plot(small_test_confs, small_test_pred_label, test_data_set['label'], network_type='small', batch=True, num_bins=15)


    # Empty the cache memory of the GPU
    del large_test_logits, small_test_logits
    torch.cuda.empty_cache()
    gc.collect()

    # Load the simplex data
    simplex_res = 0.005
    simplex = np.load('./simplex/{}.npy'.format(simplex_res))

    index_set_test = {k: [] for k in distance_function}
    set_size_test = {k: [] for k in distance_function}
    coverage_function = {k: 0 for k in distance_function}

    for name in distance_function:
        print(f'current NC score is {name}')
        final_conf_avg, final_label_avg = [0]*10000, [0]*10000
        final_conf_ip, final_label_ip = [0] * 10000, [0] * 10000
        final_conf_convex, final_label_convex, entropy_convex = [0] * 10000, [0] * 10000, [0] * 10000
        for k in range(len(large_test_probs)):
            for i in range(args.batch_size):
                current_index = (k * args.batch_size) + i

                if name == 'kl':
                    idx_kl = np.where(entropy(simplex, small_test_probs[k][i], base=2, axis=1) < calibration_score_quantile['kl'])
                    coverage_function['kl'] += (entropy(large_test_probs[k][i], small_test_probs[k][i], base=2) < calibration_score_quantile['kl'])
                    final_soft_decision_avg = sum(simplex[idx_kl]) / len(idx_kl[0])

                elif name == 'alpha':
                    idx_kl = []
                    for j in range(simplex.shape[0]):
                        if D_alpha(simplex[j], small_test_probs[k][i], alpha=args.alpha_div) < calibration_score_quantile['alpha']:
                            idx_kl.append(j)

                    # coverage_function['alpha'] += (D_alpha(large_test_probs[k][i], small_test_probs[k][i], alpha=args.alpha_div) < calibration_score_quantile['alpha'])
                    final_soft_decision_avg = sum(simplex[idx_kl]) / len(idx_kl)

                elif name == 'tv':
                    idx_kl = np.where(0.5*np.sum(np.abs(simplex - small_test_probs[k][i]), axis=1) < calibration_score_quantile['tv'])
                    # coverage_function['tv'] += (0.5*np.sum(np.abs(large_test_probs[k][i] - small_test_probs[k][i])) < calibration_score_quantile['tv'])
                    final_soft_decision_avg = sum(simplex[idx_kl]) / len(idx_kl[0])

                elif name == 'inner':
                    idx_kl = np.where(1 - np.inner(simplex, small_test_probs[k][i]) < calibration_score_quantile['inner'])
                    # coverage_function['inner'] += (1 - np.inner(large_test_probs[k][i], small_test_probs[k][i]) <  calibration_score_quantile['inner'])
                    final_soft_decision_avg = sum(simplex[idx_kl]) / len(idx_kl[0])

                elif name == 'ws':
                    idx_kl = []
                    for j in range(simplex.shape[0]):
                        if wasserstein_distance([0, 1, 2], [0, 1, 2], simplex[j], small_calibration_probs[k][i]) < calibration_score_quantile['ws']:
                            idx_kl.append(j)
                    # coverage_function['ws'] += (wasserstein_distance([0, 1, 2], [0, 1, 2], large_test_probs[k][i], small_calibration_probs[k][i]) < calibration_score_quantile['ws'])
                    final_soft_decision_avg = sum(simplex[idx_kl]) / len(idx_kl)



                # # print(simplex[idx_kl])
                #
                # '''
                #     Directly average to make the single final decision
                # '''
                # # final_soft_decision_avg = sum(simplex[idx_kl]) / len(idx_kl[0])
                # final_label_avg[current_index] = np.argmax(final_soft_decision_avg)
                # final_conf_avg[current_index] = np.max(final_soft_decision_avg)
                #
                # '''
                #     Using intersection probability to make the single final decision
                # '''
                # upper_bound = np.max(simplex[idx_kl], axis=0)
                # lower_bound = np.min(simplex[idx_kl], axis=0)
                # beta = (1 - np.sum(lower_bound)) / np.sum(upper_bound - lower_bound)
                # final_soft_decision_ip = lower_bound + beta * (upper_bound - lower_bound)
                # final_label_ip[current_index] = np.argmax(final_soft_decision_ip)
                # final_conf_ip[current_index] = np.max(final_soft_decision_ip)
                #
                # '''
                #     Using maximum Shannon entropy within credal set to make the single final decision
                # '''
                # S_entropy = 0
                # for p in simplex[idx_kl]:
                #     # Ensure that there are no zeros (avoid log(0)) and that all probabilities are valid
                #     q_without_zero = [q for q in p if q > 0]
                #     S_entropy_new = -sum(q1 * np.log2(q1) for q1 in q_without_zero)
                #     if S_entropy_new >= S_entropy:
                #         final_label_convex[current_index] = np.argmax(p)
                #         final_conf_convex[current_index] = np.max(p)






                # idx_kl_swap, idx_alpha, idx_alpha_swap = [], [], []
                #
                # for j in range(simplex.shape[0]):
                #     if D_alpha(simplex[j], small_test_probs[k][i], alpha=args.alpha_div) < calibration_score_quantile['alpha']:
                #         idx_alpha.append(j)
                #
                # coverage_function['alpha'] += (D_alpha(large_test_probs[k][i], small_test_probs[k][i], alpha=args.alpha_div) < calibration_score_quantile['alpha'])

                index_set_test[name].append(idx_kl)
                set_size_test[name].append(len(idx_kl[0]))
                #
                # index_set_test['alpha'].append(idx_alpha)
                # set_size_test['alpha'].append(len(idx_alpha))

        # label_mapping = ['contradiction', 'entailment', 'neutral']
        #
        # '''
        #     Test for directly averaging
        # '''
        # final_conf_avg = torch.tensor(final_conf_avg)
        # final_label_avg = [label_mapping[score_max] for score_max in final_label_avg]
        #
        # ECE, ACC, _ = pf.expected_calibration_error(final_conf_avg, final_label_avg, test_data_set['label'], batch=False, num_bins=15)
        # print(f'Direct average: current small ECE is {ECE * 100}%, acc is {ACC * 100}%')
        # # pf.reliability_diagram_plot(final_conf_avg, final_label_avg, test_data_set['label'], network_type='avg', batch=False, num_bins=15)
        #
        # '''
        #     Test for intersection probability
        # '''
        # final_conf_ip = torch.tensor(final_conf_ip)
        # final_label_ip = [label_mapping[score_max] for score_max in final_label_ip]
        # ECE, ACC, _ = pf.expected_calibration_error(final_conf_ip, final_label_ip, test_data_set['label'], batch=False, num_bins=15)
        # print(f'Intersection probability: current small ECE is {ECE * 100}%, acc is {ACC * 100}%')
        # # pf.reliability_diagram_plot(final_conf_ip, final_label_ip, test_data_set['label'], network_type='ip', batch=False, num_bins=15)
        #
        # '''
        #     Test for maximum Shannon entropy
        # '''
        # final_conf_convex = torch.tensor(final_conf_convex)
        # final_label_convex = [label_mapping[score_max] for score_max in final_label_convex]
        # ECE, ACC, _ = pf.expected_calibration_error(final_conf_convex, final_label_convex, test_data_set['label'], batch=False, num_bins=15)
        # print(f'Maximum Shannon entropy: current small ECE is {ECE * 100}%, acc is {ACC * 100}%')
        # # pf.reliability_diagram_plot(final_conf_convex, final_label_convex, test_data_set['label'], network_type='convex', batch=False, num_bins=15)
        #
        # pf.confidence_distribution_plot(large_test_confs, final_conf_ip, 'credal', num_bins=15)
        # pf.confidence_distribution_plot(large_test_confs, small_test_confs, 'small', num_bins=15)

    # Check the credal set efficiency
    efficiency_function = {k: [] for k in distance_function}
    print(f'Current temperature for SoftMax layer is {args.temp}')
    print(f'Current alpha divergence: {args.alpha_div} Current alpha quantile: {args.alpha_quant}')
    for name in distance_function:
        efficiency_function[name] = np.sum(set_size_test[name])
        efficiency_function[name] = efficiency_function[name] / (simplex.shape[0] * test_data_size)
        coverage_function[name] = (coverage_function[name] / test_data_size)
        print(f'{name} efficiency: {efficiency_function[name]}')
        print(f'{name} coverage: {coverage_function[name]}')

    # Plot the simplex figure
    # set_points = simplex[index_set_test['kl'][args.test_index]]
    # predict_points = small_test_prob[args.test_index]
    # true_points = large_test_prob[args.test_index]
    # cove = coverage_function['kl']
    # eff = set_size_test['kl'][args.test_index] / simplex.shape[0]
    # plot_nli_distribution(set_points, predict_points, true_points, cove, eff, args)

       # Make a new direction to save the data
        alpha_div = str(args.alpha_div)
        alpha_quant = str(args.alpha_quant)

        # os.makedirs(os.path.join("results", 'nlp_figures'), exist_ok=True)
        # with open(os.path.join("results", 'nlp_figures', "idx_set_test.pkl"), 'wb') as f:
        #     pickle.dump(index_set_test, f)
        # with open(os.path.join("results", 'nlp_figures', "set_size_test.pkl"), 'wb') as f:
        #     pickle.dump(set_size_test, f)
        # with open(os.path.join("results", 'nlp_figures', "prediction.pkl"), 'wb') as f:
        #     pickle.dump(small_test_prob, f)
        # with open(os.path.join("results", 'nlp_figures', "true.pkl"), 'wb') as f:
        #     pickle.dump(large_test_prob, f)
        # with open(os.path.join("results", 'nlp_figures', "coverage.pkl"), 'wb') as f:
        #     pickle.dump(coverage_function, f)
        # with open(os.path.join("results", 'nlp_figures', "efficiency.pkl"), 'wb') as f:
        #     pickle.dump(efficiency_function, f)




if __name__ == '__main__':
    args = parseArgs()
    main(args)