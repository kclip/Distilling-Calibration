import argparse
import torch
from torch.utils.data import random_split, Subset
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from net.miniVGG import MiniVGG
import function_image as pf
def parseArgs():

    parser = argparse.ArgumentParser(
        description="Training for calibration distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='cifar_10', help='dataset for training')
    parser.add_argument("--model", type=str, default='miniVGG', help='network architecture for training')
    parser.add_argument("--random_seed", type=int, default=1, help='random seed for training')
    parser.add_argument("--epoch", type=int, default=15, help='epoch for training')
    parser.add_argument("--iteration", type=int, default=100, help='iteration for training')
    parser.add_argument("--first_milestone", type=int, default=30, help='first learning rate change')
    parser.add_argument("--second_milestone", type=int, default=60, help='second learning rate change')
    parser.add_argument("--lr", type=float, default=0.001, help='learning rate for training')
    parser.add_argument("--momentum", type=float, default=0.9, help='momentum for training')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help='weight decay for training')
    parser.add_argument("--batch_size", type=int, default=128, help='batch size for training')
    parser.add_argument("--optimizer", type=str, default='adam', help='optimizer for training')


    return parser.parse_args()

def main(args):

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    torch.manual_seed(1)

    cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'CUDA is {cuda} on {device}')

    ''' load CIFAR-10 data set '''

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Download and load the training data
    whole_training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

    # Filter the data set with first three classes
    train_indices = [idx for idx, (_, label) in enumerate(whole_training_set) if label < 3]

    training_data_subset = Subset(whole_training_set, train_indices)

    training_data, validation_data = random_split(training_data_subset, [13500, 1500])
    calibration_data, _ = random_split(validation_data, [500, 1000])

    # Construct the data loader
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    calibration_data_loader = torch.utils.data.DataLoader(calibration_data, batch_size=len(calibration_data), shuffle=True)

    # Initialize the model
    if args.model == 'miniVGG':
        model = MiniVGG()

    model.to(device)
    print(model)

    if args.optimizer == "sgd":
        opt_params = model.parameters()
        optimizer = torch.optim.SGD(opt_params,
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=False)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1)

    model.train()
    for epoch in range(args.epoch):
        scheduler.step()
        for batch_idx, (data, labels) in enumerate(training_data_loader):
            data, labels = data.to(device), labels.to(device)

            logits = model(data)
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='sum')

            optimizer.zero_grad()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 2)
            loss.backward()
            optimizer.step()
        print(f'epoch is {epoch}, loss value is {loss.item()}')

    torch.save(model, './trained_model/{}_iteration_{}.pt'.format(args.model, args.iteration))


    '''
            model test
    '''

    # Download and load the test data
    test_data_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_indices = [idx for idx, (_, label) in enumerate(test_data_set) if label < 3]
    test_data_subset = Subset(test_data_set, test_indices)
    test_data_loader = torch.utils.data.DataLoader(test_data_subset, batch_size=len(test_data_subset), shuffle=False)

    # Load the trained model
    large_model = torch.load('./trained_model/resnet_18.pt')
    small_model = torch.load('./trained_model/{}_iteration_{}.pt'.format(args.model, args.iteration))

    '''
                Do the TS to calibrate the large-scale model
    '''
    ece_val = 10 ** 7
    T_opt_ece = 1.0
    T = 0.1

    for i in range(40):
        temperature = T
        for batch_idx, (data, labels) in enumerate(calibration_data_loader):
            data, labels = data.to(device), labels.to(device)
            logits = large_model(data) / temperature
            large_test_prob = torch.nn.functional.softmax(logits.data, dim=1)
            large_test_pred_confidence, large_test_pred_label = torch.max(large_test_prob.data, dim=1)
            ECE = pf.expected_calibration_error(large_test_pred_confidence, large_test_pred_label, labels,
                                                num_bins=15) * 100
            accuracy = 100 * torch.sum(large_test_pred_label.eq(labels)) / len(calibration_data)
            print(f'Current large model with {T} temperature on calibration data set: ECE is {ECE}, ACC is {accuracy}')

        if ece_val > ECE:
            T_opt_ece = T
            ece_val = ECE
        T += 0.1

    print(f'The best Temperature for TS is {T_opt_ece}')

    '''
            Evaluate the small trained model
    '''
    small_model.eval()
    large_model.eval()

    with torch.no_grad():
        T_opt_ece = 1.8

        for batch_idx, (data, labels) in enumerate(test_data_loader):
            data, labels = data.to(device), labels.to(device)
            logits = large_model(data) / T_opt_ece
            large_test_prob = torch.nn.functional.softmax(logits.data, dim=1)
            large_test_pred_confidence, large_test_pred_label = torch.max(large_test_prob.data, dim=1)

            ECE = pf.expected_calibration_error(large_test_pred_confidence, large_test_pred_label, labels,
                                                num_bins=15) * 100
            accuracy = 100 * torch.sum(large_test_pred_label.eq(labels)) / len(test_data_subset)
            print(f'Large model on test data set: ECE is {ECE}, ACC is {accuracy}')

        for batch_idx, (data, labels) in enumerate(test_data_loader):
            data, labels = data.to(device), labels.to(device)
            logits = small_model(data)
            small_test_prob = torch.nn.functional.softmax(logits.data, dim=1)
            small_test_pred_confidence, small_test_pred_label = torch.max(small_test_prob.data, dim=1)

            ECE = pf.expected_calibration_error(small_test_pred_confidence, small_test_pred_label, labels, num_bins=15) * 100
            accuracy = 100 * torch.sum(small_test_pred_label.eq(labels)) / len(test_data_subset)
            print(f'Small model on test data set: ECE is {ECE}, ACC is {accuracy}')


if __name__ == '__main__':
    args = parseArgs()
    main(args)