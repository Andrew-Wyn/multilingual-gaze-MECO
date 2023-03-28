import sys
import os
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

import argparse
import numpy as np
from transformers import set_seed
from gaze.utils import Config, minMaxScaling
from gaze.dataset import GazeDataset
import torch
import json
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


def compute_mean_baseline(args, dataset, id):

    print(f"--- {id} ---")

    # Dataset
    d = GazeDataset(cf, None, dataset)
    d.load_data()

    flat_targets = [item for sublist in d.targets for item in sublist]

    target = np.asarray(flat_targets, dtype=np.float32)

    # do the cross validation
    # compute the mean-baseline estimators
    shuffled_ids = shuffle(range(target.shape[0]), random_state=cf.seed)
    target = target[shuffled_ids]

    train_target = target[:int(target.shape[0]*0.9)]
    test_target = target[int(target.shape[0]*0.9):]

    scaler = MinMaxScaler(feature_range=[0, cf.feature_max])

    train_target = scaler.fit_transform(train_target)
    test_target = scaler.transform(test_target)

    means = np.mean(train_target, axis=0)

    predicted = np.array([means,]*test_target.shape[0])

    # compute the mse and mae
    mae_loss = torch.nn.L1Loss(reduction="sum")
    mse_loss = torch.nn.MSELoss(reduction="mean")

    losses = dict()

    for feat_i in range(target.shape[1]):
        feat_mae_loss = mae_loss(torch.tensor(test_target[:, feat_i]), torch.tensor(predicted[:, feat_i]))
        feat_mse_loss = mse_loss(torch.tensor(test_target[:, feat_i]), torch.tensor(predicted[:, feat_i]))
        losses[f"mae_loss{feat_i}"] = feat_mae_loss.item()/test_target.shape[0]
        losses[f"mse_loss{feat_i}"] = feat_mse_loss.item()
    
    losses["mae_loss_all"] = mae_loss(torch.tensor(test_target), torch.tensor(predicted)).item()/(test_target.shape[0] * test_target.shape[1])
    losses["mse_loss_all"] = mse_loss(torch.tensor(test_target), torch.tensor(predicted)).item()

    with open(f"{args.output_dir}/mean_baseline_results_{id}.json", 'w') as f:
        json.dump(losses, f)

    print("--- end ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the mean-baseline performance over a Gaze dataset')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script')
    parser.add_argument('-o', '--output-dir', dest='output_dir', action='store',
                        help=f'Relative path of output directory')
    
    # Read the script's argumenents
    args = parser.parse_args()
    config_file = args.config_file

    # Load the .json configuration file
    cf = Config.load_json(config_file)

    # set seed
    set_seed(cf.seed)

    # check if the output directory exists, if not create it!
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    compute_mean_baseline(args, "data/datasets/en/all_mean_dataset.csv", "en_all")
    compute_mean_baseline(args, "data/datasets/it/all_mean_dataset.csv", "it_all")
    compute_mean_baseline(args, "data/datasets/it/cluster_0_dataset.csv", "it_0")
    compute_mean_baseline(args, "data/datasets/it/cluster_1_dataset.csv", "it_1")
    compute_mean_baseline(args, "data/datasets/it/cluster_2_dataset.csv", "it_2")
    compute_mean_baseline(args, "data/datasets/sp/all_mean_dataset.csv", "sp_all")
    compute_mean_baseline(args, "data/datasets/sp/cluster_0_dataset.csv", "sp_0")
    compute_mean_baseline(args, "data/datasets/sp/cluster_1_dataset.csv", "sp_1")
    compute_mean_baseline(args, "data/datasets/sp/cluster_2_dataset.csv", "sp_2")
    compute_mean_baseline(args, "data/datasets/ge/all_mean_dataset.csv", "ge_all")
    compute_mean_baseline(args, "data/datasets/ge/cluster_0_dataset.csv", "ge_0")
    compute_mean_baseline(args, "data/datasets/ge/cluster_1_dataset.csv", "ge_1")
    compute_mean_baseline(args, "data/datasets/ge/cluster_2_dataset.csv", "ge_2")