import os
import argparse
import torch
from torch.utils.data import random_split, Subset
import numpy as np
import torchvision
from torchvision import transforms
from net.resnet import resnet110, resnet18
import matplotlib.pyplot as plt
from scipy.stats import entropy
import function_image as pf
from laplace import Laplace
import cvxpy as cp


def parseArgs():

    parser = argparse.ArgumentParser(
        description="Distilling Calibration via Conformalized Credal Inference: CIFAR-10 task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='cifar_10', help='dataset for training')
    parser.add_argument("--model", type=str, default='resnet_18', help='network architecture for training')
    parser.add_argument("--random_seed", type=int, default=1, help='random seed for training')
    parser.add_argument("--epoch", type=int, default=20, help='epoch for training')
    parser.add_argument("--first_milestone", type=int, default=150, help='first learning rate change')
    parser.add_argument("--second_milestone", type=int, default=250, help='second learning rate change')
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate for training')
    parser.add_argument("--momentum", type=float, default=0.9, help='momentum for training')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help='weight decay for training')
    parser.add_argument("--batch_size", type=int, default=128, help='batch size for training')
    parser.add_argument("--optimizer", type=str, default='sgd', help='optimizer for training')
    parser.add_argument("--test_index", type=int, default=578, help='data index for plotting')
    parser.add_argument("--alpha_div", type=float, default=0.9, help='alpha for alpha_divergence')
    parser.add_argument("--alpha_quant", type=float, default=0.1, help='alpha quantile for NC score')
    parser.add_argument("--temp", type=float, default=1, help='temperature for softmax layer')
    parser.add_argument("--small_model", type=str, default='miniVGG', help='name for small model')
    parser.add_argument("--iteration", type=int, default='1000', help='iteration for small model')
    parser.add_argument("--la_approx", type=bool, default=False, help='if using Laplace approx. for small model')
    parser.add_argument("--cvx", type=bool, default=False, help='if using CVX for single predictive distribution')

    return parser.parse_args()

def D_alpha(p, q, alpha):
    '''
        Alpha divergence for measuring discrepancy between two distribution p and q
        :param p: p is the large-scale model soft decision or sampling distributions from the simplex P
        :param q: q is the small-scale model soft decision
        :param alpha: alpha is the parameter to control the divergence function, one important hyperparameter, please take care when choosing
        :param swap: this function provides also a way for swapping p and q, since the divergence itself is asymetric
        :return: return a distance value for two distributions q and p, given the alpha
    '''

    divergence = np.sum(p ** alpha * q ** (1 - alpha))
    return 1 / (alpha * (alpha - 1)) * (divergence - 1)


def compute_quantile(scores, alpha):
    '''
        This function computes the quantile of a distribution given a scores array
        :param scores: a vector of the values drawn from the distribution
        :param alpha: alpha in this function is the 1-alpha quantile among the provided sampled data vector, alpha also can be viewed as the user defined level or target miscoverage rate
        :return: return the 1-alpha quantile element among the provided sampled data vector
    '''

    n = len(scores)
    return np.quantile(scores, np.ceil((n+1)*(1-alpha))/n, method="inverted_cdf")

def plot_nli_distribution(dist, prediction, true, cvg, efficiency, args, labels=['Airplane', 'Automobile', 'Bird']):

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
    point = true[0] * vertices[0] + true[1] * vertices[1] + true[2] * vertices[2]
    ax.plot(point[0], point[1], 's', color='red', alpha=1, markersize=3)

    for index in range(dist.shape[0]):
        point = dist[index][0] * vertices[0] + dist[index][1] * vertices[1] + dist[index][2] * vertices[2]
        ax.plot(point[0], point[1], 'o', color='green', alpha=0.3, markersize=0.4)

    plt.text(0.3, -0.1, 'Per-data inefficiency:{}%'.format(round(efficiency*100, 2)))

    # Set limits and aspect
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')  # Hide the axes
    plt.savefig('./simplex_figures/test_data_{}.jpeg'.format(args.test_index), dpi=1000)

    plt.show()

def main(args):

    distance_functions = ['kl']

    overall_coverage = {k: [] for k in distance_functions}
    overall_efficiency = {k: [] for k in distance_functions}

    for ran in range(1):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        random_seed = 1
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
        # Please note that, to avoid the distributions shift between calibration data set and test data set, do not use data augmentation for calibration data set

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        # Download and load the training data
        whole_training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

        # Filter the data set with first three classes
        train_indices = [idx for idx, (_, label) in enumerate(whole_training_set) if label < 3]

        training_data_subset = Subset(whole_training_set, train_indices)

        training_data, validation_data = random_split(training_data_subset, [13500, 1500])
        calibration_data, _ = random_split(validation_data, [1000, 500])

        # Construct the data loader
        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
        calibration_data_loader = torch.utils.data.DataLoader(calibration_data, batch_size=len(calibration_data), shuffle=False)

        '''
                model test
        '''

        # Download and load the test data
        test_data_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_indices = [idx for idx, (_, label) in enumerate(test_data_set) if label < 3]
        test_data_subset = Subset(test_data_set, test_indices)
        # test_data_subset, _ = random_split(test_data_subset, [1500, 1500])


        '''
            Test for splitting test data set to one half
        '''
        # calibration_data, test_data = random_split(test_data_subset, [1500, 1500])
        # calibration_data_loader = torch.utils.data.DataLoader(calibration_data, batch_size=len(calibration_data), shuffle=True)
        # test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)

        # Full batch for test data
        test_data_loader = torch.utils.data.DataLoader(test_data_subset, batch_size=len(test_data_subset), shuffle=False)

        # Load the trained model
        large_model = torch.load('./trained_model/resnet_18.pt', map_location=device)
        total_params_large = sum(p.numel() for p in large_model.parameters() if p.requires_grad)

        # small_model = torch.load('./trained_model/{}_iteration_{}.pt'.format(args.small_model, args.iteration), map_location=device)
        small_model = torch.load('./trained_model/miniVGG_30.pt', map_location=device)
        total_params_small = sum(p.numel() for p in small_model.parameters() if p.requires_grad)

        print(f'Number of parameter for large model {total_params_large}')
        print(f'Number of parameter for small model {total_params_small}')

        '''
                Evaluate the large trained model
        '''
        large_model.eval()
        with torch.no_grad():

            '''
                        Do the TS to calibrate the large-scale model
                        To guarantee the calibration performance of the large-scale model, or keep a relatively huge gap between the large-scale one and the small-scale one
                        Please do the evaluation for these two models first to check their calibration performance in terms of ECE
                        If the large-scale model can not satisfy the above assumption, please use trainable or post-processing methods to re-calibrate the decisions of the models
                        Following: We give a example to show how to re-calibrate the large-scale models decision by using the typical post-processing methods, namely Temperature Scaling
                        For the details of the TS, please refer to the original paper 'Guo calibration'
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
            # T_opt_ece = 2.1

            # Information from calibration data set
            for batch_idx, (data, labels) in enumerate(calibration_data_loader):
                data, labels = data.to(device), labels.to(device)
                logits = large_model(data) / T_opt_ece
                large_calibration_prob = torch.nn.functional.softmax(logits.data, dim=1)
                large_calibration_pred_confidence, large_calibration_pred_label = torch.max(large_calibration_prob.data, dim=1)

                ECE = pf.expected_calibration_error(large_calibration_pred_confidence, large_calibration_pred_label, labels, num_bins=15) * 100
                accuracy = 100 * torch.sum(large_calibration_pred_label.eq(labels)) / len(calibration_data)
                # print(f'Large model on calibration data set: ECE is {ECE}, ACC is {accuracy}')



            # Information from test data set
            for batch_idx, (data, labels) in enumerate(test_data_loader):
                data, labels = data.to(device), labels.to(device)
                logits = large_model(data) / T_opt_ece
                large_test_prob = torch.nn.functional.softmax(logits.data, dim=1)
                large_test_pred_confidence, large_test_pred_label = torch.max(large_test_prob.data, dim=1)

                # Calculate the accuracy for large model
                ECE = pf.expected_calibration_error(large_test_pred_confidence, large_test_pred_label, labels, num_bins=15) * 100
                accuracy = 100 * torch.sum(large_test_pred_label.eq(labels)) / len(large_test_pred_label)
                print(f'Large model on test data set: ECE is {ECE}, ACC is {accuracy}')

                # pf.reliability_diagram_plot(large_test_pred_confidence, large_test_pred_label, labels, ECE, accuracy, 'large', num_bins=15)


            # Use detach before cpu to avoid gradient copied
            large_calibration_prob = large_calibration_prob.detach().cpu().numpy()
            large_test_prob = large_test_prob.detach().cpu().numpy()

        '''
            Evaluate the small trained model
        '''
        small_model.eval()
        with torch.no_grad():

            # Information from calibration data set
            for batch_idx, (data, labels) in enumerate(calibration_data_loader):
                data, labels = data.to(device), labels.to(device)
                logits = small_model(data) / args.temp
                small_calibration_prob = torch.nn.functional.softmax(logits.data, dim=1)
                small_calibration_pred_confidence, small_calibration_pred_label = torch.max(small_calibration_prob.data, dim=1)

                ECE = pf.expected_calibration_error(small_calibration_pred_confidence, small_calibration_pred_label,
                                                    labels, num_bins=15) * 100
                accuracy = 100 * torch.sum(small_calibration_pred_label.eq(labels)) / len(calibration_data)
                print(f'Small model on calibration data set: ECE is {ECE}, ACC is {accuracy}')

            # Information from test data set
            for batch_idx, (data, labels) in enumerate(test_data_loader):
                data, labels = data.to(device), labels.to(device)
                logits = small_model(data) / args.temp
                small_test_prob = torch.nn.functional.softmax(logits.data, dim=1)
                small_test_pred_confidence, small_test_pred_label = torch.max(small_test_prob.data, dim=1)

                # Calculate the accuracy for large model
                ECE = pf.expected_calibration_error(small_test_pred_confidence, small_test_pred_label, labels, num_bins=15) * 100
                accuracy = 100 * torch.sum(small_test_pred_label.eq(labels)) / len(small_test_pred_label)
                print(f'Small model on test data set: ECE is {ECE}, ACC is {accuracy}')

                # pf.reliability_diagram_plot(small_test_pred_confidence, small_test_pred_label, labels, ECE, accuracy, 'small', num_bins=15)



            '''
                Below, we apply one of the low-complexity Bayesian learning way to re-calibrate the small-scale model soft decision, namely Laplace approximation
                For the details and Laplace package, you can refer the original paper 'Laplace approximation' for the details
                In this task, due to the high computational costs for applying the Laplace approximation
                We just apply the Laplace approximation for the last hidden layer, the size of the last hidden layer totally depends on the pre-trained models the network architecture
                For the covariance approximation, to reduce the computational costs, we just use the diagonal not the full matrix
                
            '''
            if args.la_approx is True:
                la = Laplace(small_model, "classification",
                             subset_of_weights="last_layer",
                             hessian_structure="diag")
                la.fit(training_data_loader)
                la.optimize_prior_precision(
                    method="gridsearch",
                    pred_type="glm",
                    link_approx="probit",
                    val_loader=calibration_data_loader
                )

                pred_la = la(data, pred_type="glm", link_approx="probit")

                small_test_pred_confidence_la, small_test_pred_label_la = torch.max(pred_la, dim=1)

                # Calculate the ECE and accuracy for large model
                ECE_la = pf.expected_calibration_error(small_test_pred_confidence_la, small_test_pred_label_la, labels,
                                                    num_bins=15) * 100
                accuracy_la = 100 * torch.sum(small_test_pred_label_la.eq(labels)) / len(small_test_pred_label_la)
                print(f'Small model with Laplace on test data set: ECE is {ECE_la}, ACC is {accuracy_la}')

                # pf.reliability_diagram_plot(small_test_pred_confidence_la, small_test_pred_label_la, labels, ECE_la, accuracy_la, 'la', num_bins=15)


            # Use detach before cpu to avoid gradient copied
            small_calibration_prob = small_calibration_prob.detach().cpu().numpy()
            small_test_prob = small_test_prob.detach().cpu().numpy()

        '''
                Calculate the alpha quantile based on the calibration data set 
        '''

        calibration_scores = {k: [] for k in distance_functions}
        calibration_scores_quantile = {k: [] for k in distance_functions}

        for i in range(len(calibration_data)):

            calibration_scores['kl'].append(entropy(large_calibration_prob[i], small_calibration_prob[i], base=2))
            # calibration_scores['alpha'].append(D_alpha(large_calibration_prob[i], small_calibration_prob[i], alpha=args.alpha_div))


        # Calculate the alpha quantile for each NC score set
        for nc in distance_functions:
            calibration_scores_quantile[nc] = compute_quantile(calibration_scores[nc], args.alpha_quant)
            print(f'NC function: {nc}, {nc}\'s quantile: {calibration_scores_quantile[nc]}')


        # Load the simplex data, pre-processing before on the other python file, please check
        simplex_res = 0.005
        simplex = np.load('./simplex/{}.npy'.format(simplex_res))

        index_set_test = {k: [] for k in distance_functions}
        set_size_test = {k: [] for k in distance_functions}

        # Check the credal set coverage
        coverage_function = {k: 0 for k in distance_functions}

        '''
            Finally, we will show how to use the information of the set predictor to make a usable single hard decision
            In this paper, we show several way to make the single hard decision
            First, we apply the most straightforward way, namely ensemble as the Bayesian learning do, i.e., directly average all of the distributions within the set
            Second, we apply the intersection probability theory referred from the credal wrapper paper, which uses the boundary information of the credal set, i.e., lower and upper bounds for each classes
            Then, since we do not have any knowledge regarding the distribution of the labels, the only way is to treat each classes equally likely, i.e., same weight or relatively distance between lower and upper bounds
            The third way is maximum Shannon entropy, which aims at finding the distributions within the credal set, who has the maximum Shannon entropy, if you have a good knowledge of the Shannon entropy,
            you will find that the higher the Shannon entropy is, the lower the confidence level or the higher the uncertainty, which can help the small-scale model to make the most conservative decisions given the corresponding credal set,
            also, please note that if you are handling the high dimension cases or the simplex is too huge due to somehow high inefficiency, you can directly use the CVX tools from convex optimization
            to solve this maximum problem, since the original statistic theory of the credal set already shown that the credal set itself is strictly convex.
        '''

        for name in distance_functions:
            final_conf_avg, final_label_avg = [0] * 3000, [0] * 3000
            final_conf_ip, final_label_ip = [0] * 3000, [0] * 3000
            final_conf_convex, final_label_convex, entropy_convex = [0] * 3000, [0] * 3000, [0] * 3000
            print(f'\n Current schemes is {name} \n')

            for i in range(len(small_test_pred_label)):

                uni_weight = torch.ones(3).cuda()
                small_test_prob = torch.tensor(small_test_prob).cuda()
                for i in range(200):
                    con_weight = small_test_prob[i]

                    sampling_dist = torch.distributions.Dirichlet(con_weight).sample((1000,)).cuda()

                    log_density_uni = torch.distributions.Dirichlet(uni_weight).log_prob(sampling_dist).cuda()
                    log_density_con = torch.distributions.Dirichlet(con_weight).log_prob(sampling_dist).cuda()

                    # explicitly compute stabilized log weights
                    log_weights = log_density_uni - log_density_con
                    log_weights -= torch.max(log_weights)  # explicitly stabilize
                    weights = torch.exp(log_weights)
                    tvd_values = 0.5 * torch.sum(torch.abs(sampling_dist - small_test_prob[i]), dim=1)
                    indicator = (tvd_values <= calibration_scores_quantile['kl']).int().cuda()
                    # print(indicator.sum)

                    print(100 * (indicator * weights).mean().item())



                if name == 'kl':
                    idx_kl = np.where(entropy(simplex, small_test_prob[i], base=2, axis=1) < calibration_scores_quantile['kl'])
                    coverage_function['kl'] += (entropy(large_test_prob[i], small_test_prob[i], base=2) < calibration_scores_quantile['kl'])

                    index_set_test['kl'].append(idx_kl)
                    set_size_test['kl'].append(len(idx_kl[0]))
                else:
                    idx_alpha = []

                    for j in range(simplex.shape[0]):
                        if D_alpha(simplex[j], small_test_prob[i], alpha=args.alpha_div) < calibration_scores_quantile['alpha']:
                            idx_alpha.append(j)

                    coverage_function['alpha'] += (D_alpha(large_test_prob[i], small_test_prob[i], alpha=args.alpha_div) < calibration_scores_quantile['alpha'])

                    index_set_test['alpha'].append(idx_alpha)
                    set_size_test['alpha'].append(len(idx_alpha))

                    idx_kl = idx_alpha

                current_index = i

                '''
                    Directly average to make the single final decision
                '''
                if name == 'kl':
                    final_soft_decision_avg = sum(simplex[idx_kl]) / len(idx_kl[0])
                else:
                    final_soft_decision_avg = sum(simplex[idx_kl]) / len(idx_kl)
                final_label_avg[current_index] = np.argmax(final_soft_decision_avg)
                final_conf_avg[current_index] = np.max(final_soft_decision_avg)

                '''
                    Using intersection probability to make the single final decision
                '''
                upper_bound = np.max(simplex[idx_kl], axis=0)
                lower_bound = np.min(simplex[idx_kl], axis=0)
                if np.sum(upper_bound - lower_bound) == 0:
                    final_soft_decision_ip = simplex[idx_kl][0]
                else:
                    beta = (1 - np.sum(lower_bound)) / np.sum(upper_bound - lower_bound)
                    final_soft_decision_ip = lower_bound + beta * (upper_bound - lower_bound)
                final_label_ip[current_index] = np.argmax(final_soft_decision_ip)
                final_conf_ip[current_index] = np.max(final_soft_decision_ip)

                '''
                    Using maximum Shannon entropy within credal set to make the single final decision
                '''
                S_entropy = 0
                for p in simplex[idx_kl]:
                    # Ensure that there are no zeros (avoid log(0)) and that all probabilities are valid
                    q_without_zero = [q for q in p if q > 0]
                    S_entropy_new = -sum(q1 * np.log2(q1) for q1 in q_without_zero)
                    if S_entropy_new >= S_entropy:
                        final_label_convex[current_index] = np.argmax(p)
                        final_conf_convex[current_index] = np.max(p)

            '''
                  Test for directly averaging
            '''

            final_conf_avg = torch.tensor(final_conf_avg).to(device)
            final_label_avg = torch.tensor(final_label_avg).to(device)

            ECE = pf.expected_calibration_error(final_conf_avg, final_label_avg, labels,
                                                num_bins=15) * 100
            accuracy = 100 * torch.sum(final_label_avg.eq(labels)) / len(final_label_avg)
            print(f'Direct average: current small ECE is {ECE}%, acc is {accuracy}%')

            '''
                Test for intersection probability
            '''

            final_conf_ip = torch.tensor(final_conf_ip).to(device)
            final_label_ip = torch.tensor(final_label_ip).to(device)

            ECE = pf.expected_calibration_error(final_conf_ip, final_label_ip, labels,
                                                num_bins=15) * 100
            accuracy = 100 * torch.sum(final_label_ip.eq(labels)) / len(final_label_ip)
            print(f'Intersection probability: current small ECE is {ECE}%, acc is {accuracy}%')

            '''
                Test for maximum Shannon entropy
            '''
            final_conf_convex = torch.tensor(final_conf_convex).to(device)
            final_label_convex = torch.tensor(final_label_convex).to(device)

            ECE = pf.expected_calibration_error(final_conf_convex, final_label_convex, labels,
                                                num_bins=15) * 100
            accuracy = 100 * torch.sum(final_label_convex.eq(labels)) / len(final_label_convex)

            print(f'Maximum Shannon entropy: current small ECE is {ECE}%, acc is {accuracy}%')


        # Check the credal set efficiency
        efficiency_function = {k: [] for k in distance_functions}
        for name in distance_functions:
            efficiency_function[name] = np.sum(set_size_test[name])
            efficiency_function[name] = efficiency_function[name] / (simplex.shape[0] * len(test_data_subset))
            print(f'{name} efficiency: {100*efficiency_function[name]}%')

            overall_coverage[name].append(coverage_function[name] / len(test_data_subset))
            overall_efficiency[name].append(efficiency_function[name])
            print(f'{name} coverage: {100*coverage_function[name] / len(test_data_subset)}%')



if __name__ == '__main__':
    args = parseArgs()
    main(args)