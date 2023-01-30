import os
import json
import numpy as np

def collect_losses(root_dir):

    depth = 2

    seeds = [123, 456]

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