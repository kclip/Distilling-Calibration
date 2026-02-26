import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = 'DejaVu Serif'
plt.rcParams['font.size'] = 12

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomData(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

num_bins=15
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'

def _bin_initializer(bin_dict, num_bins=15):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0
def _populate_bins(confs, preds, labels, batch, num_bins=15):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    if batch is False:
        for i in range(0, num_test_samples):
            confidence = confs[i]
            prediction = preds[i]
            label = labels[i]
            binn = int(math.ceil(((num_bins * confidence) - 1)))
            if binn == num_bins:
                binn = binn - 1
            bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
            bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
            bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)
    else:
        for i in range(0, num_test_samples):
            confidence = confs[i]
            index = i // len(preds[0])
            index_c = i % len(preds[0])
            # print(index)
            prediction = preds[index][index_c]
            label = labels[i]
            binn = int(math.ceil(((num_bins * confidence) - 1)))
            if binn == num_bins:
                binn = binn - 1
            bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
            bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
            bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(bin_dict[binn][COUNT])
    return bin_dict


def expected_calibration_error(confs, preds, labels, batch=False, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, batch, num_bins)
    num_samples = len(labels)
    ece = 0
    acc = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
        acc += (bin_accuracy * bin_count) / num_samples
    return ece, acc, bin_dict

def reliability_diagram_plot(confs, preds, labels, network_type, batch=False, num_bins=15):

    ece, acc, bin_dict = expected_calibration_error(confs, preds, labels, batch, num_bins)

    bar_width = 1 / num_bins
    bbins = np.linspace(bar_width / 2, 1 - bar_width / 2, num_bins)
    x = np.linspace(bar_width / 2, 1 - bar_width / 2, num_bins)
    bar_width = 1 / num_bins

    ECE = torch.round(ece*100, decimals=3).cpu()
    ece = round(ECE.numpy().tolist(), 3)

    accuracy = round(acc*100, 3)

    plt.figure()
    left, width = 0.1, 0.8
    bottom, height = 0.1, 0.1
    bottom_h = bottom + height + 0.05
    line1 = [left, bottom, width, 0.23]
    line2 = [left, 0.4, width, 0.5]
    ax1 = plt.axes(line2) #upper
    ax2 = plt.axes(line1) #below

    ax1.grid(True, linestyle='dashed', alpha=0.5)
    ax1.set_xlim(0.4, 1)
    bin_conf, bin_acc, bin_count = [0] * num_bins, [0] * num_bins, [0] * num_bins
    for i in range(num_bins):
        bin_conf[i], bin_acc[i], bin_count[i] = bin_dict[i][BIN_CONF], bin_dict[i][BIN_ACC], bin_dict[i][COUNT]
        bin_count[i] = bin_count[i] / len(labels)
    converted_bin_conf = []
    for item in bin_conf:
        if isinstance(item, torch.Tensor):
            converted_bin_conf.append(item.item())  # Convert the tensor to a Python float
        else:
            converted_bin_conf.append(item)

    # print(f'conf is {converted_bin_conf}, acc is {bin_acc}, count is {bin_count}')
    # print(f'count is {bin_count}')

    ax1.set_ylabel('Test Accuracy / Test Confidence')
    ax1.bar(x, converted_bin_conf, bar_width, align='center', facecolor='r', edgecolor='black', label='Gap', hatch='/', alpha=0.3)
    ax1.bar(x, bin_acc, bar_width, align='center', facecolor='b', edgecolor='black', label='Outputs', alpha=0.75)
    ax1.text(0.7, 0.25, r'ECE={}'.format(ece), fontsize=16, bbox=dict(facecolor='lightskyblue', alpha=0.9))
    ax1.text(0.7, 0.1, r'Acc={}'.format(accuracy), fontsize=16, bbox=dict(facecolor='lightskyblue', alpha=0.9))
    ax1.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=3)
    ax1.legend()
    if network_type == 'large':
        ax1.set_title('Large-Scale')
    elif network_type == 'small':
        ax1.set_title('Small-Scale')
    elif network_type == 'avg':
        ax1.set_title('Credal Avg.')
    elif network_type == 'ip':
        ax1.set_title('Credal Inter. Prob.')
    elif network_type == 'convex':
        ax1.set_title('Credal Max H(p)')

    ax2.set_xlabel('Confidence')
    ax2.bar(bbins, bin_count, bar_width, align='center', facecolor='blue', edgecolor='black', label='Gap', alpha=0.7)
    ax2.grid(True, linestyle='dashed', alpha=0.5)
    ax2.set_xlim(0.4, 1)
    ax2.set_ylabel('Sampling frequency')
    ax2.set_ylim(0, 1)
    plt.savefig(f'./{network_type}.jpeg', dpi=1000)
    plt.show()

def confidence_distribution_plot(confidence_id, confidence_ood, network_type, num_bins=15):
    confidence_id, confidence_ood = confidence_id.cpu(), confidence_ood.cpu()
    bins = np.linspace(0, 1, num_bins + 1)
    total_id = len(confidence_id)
    total_ood = len(confidence_ood)
    nsamples_each_interval_id = []
    nsamples_each_interval_ood = []
    for i in range(0, num_bins):
        temp_id = np.where((bins[i] <= confidence_id) & (confidence_id < bins[i + 1]))
        nsamples_each_interval_id.append((len(temp_id[0]) / total_id) )

        temp_ood = np.where((bins[i] <= confidence_ood) & (confidence_ood < bins[i + 1]))
        nsamples_each_interval_ood.append((len(temp_ood[0]) / total_ood))
    bar_width = 1 / num_bins
    bbins = np.linspace(bar_width / 2, 1 - bar_width / 2, num_bins)

    plt.figure()

    plt.bar(bbins, nsamples_each_interval_id, bar_width, align='center', facecolor='blue', edgecolor='black',
            label='large-scale model', alpha=0.4)
    if network_type == 'small':
        plt.bar(bbins, nsamples_each_interval_ood, bar_width, align='center', facecolor='red', edgecolor='black',
                label='small-scale model', alpha=0.4)
    elif network_type == 'credal':
        plt.bar(bbins, nsamples_each_interval_ood, bar_width, align='center', facecolor='red', edgecolor='black',
            label='CD-CI', alpha=0.4)
    plt.xlim(0.4, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='dashed', alpha=0.5)
    plt.xlabel('Confidence')
    plt.ylabel('Sampling frequency')
    plt.legend(loc='upper left')

    plt.savefig(f'./{network_type}.jpeg', dpi=1000)
    plt.show()

