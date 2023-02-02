import os
import json
import numpy as np

def collect_losses(root_dir):

    depth = 2

    seeds = [123, 456, 789]

    dict_res = dict() # {"dataset_name" : [means stats, var stats]}

    # collect ts losses

    for seed_ in seeds:
        seed_rootdir = f"{root_dir}/{seed_}"

        for subdir, dirs, files in os.walk(seed_rootdir):
            if subdir[len(seed_rootdir):].count(os.sep) < depth:
                for file_ in files:
                    if file_ == "finetuning_results.json":
                        with open(subdir + "/" + file_) as f:
                            d = json.load(f)
                            losses_ts = list(d["losses_ts"].values())

                        dataset_name = subdir.split(os.sep)[-1]
                        if dataset_name in dict_res.keys():
                            dict_res[dataset_name].append(losses_ts)
                        else:
                            dict_res[dataset_name] = [losses_ts]

    # compute mean and std for each losses

    for dataset, losses_list in dict_res.items():
        means = []
        stds = []

        losses_len = len(losses_list[0])
        for i in range(losses_len):
            losses_i = []
            for losses in losses_list:
                losses_i.append(losses[i])
            means.append(np.mean(losses_i))
            stds.append(np.std(losses_i))

        dict_res[dataset] = [means, stds]

    return dict_res


def avg_losses_over_datasets(datasets_losses):
    avg_losses_dts = None

    num_dts = 0

    for dataset, mean_std in datasets_losses.items():
        means = mean_std[0]

        if avg_losses_dts is None:
            avg_losses_dts = np.array(means)
        else:
            avg_losses_dts += np.array(means)

        num_dts += 1

    return avg_losses_dts / num_dts

if __name__ == "__main__":

    print("pretraining losses ...")

    pretrain_res = collect_losses("finetuning/pretraining")

    print(pretrain_res)

    print()
    print("---")
    print()

    print("not pretraining losses ...")

    notpretraining_res = collect_losses("finetuning/notpretraining")

    print(notpretraining_res)

    print()
    print("============================")
    print()

    print("average pretraining losses ...")

    avg_losses_dts_pretrain = avg_losses_over_datasets(pretrain_res)

    print(avg_losses_dts_pretrain)

    print()
    print("---")
    print()

    print("average not pretraining losses ...")

    avg_losses_dts_notpretraining = avg_losses_over_datasets(notpretraining_res)

    print(avg_losses_dts_notpretraining)

    print()
    print("---")
    print()

    print("difference btw pretrained and notpretrained avg losses ...")

    print(avg_losses_dts_pretrain - avg_losses_dts_notpretraining)