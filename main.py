import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_senteces_from_data(data):
    word_func = lambda s: [w for w in s["ia"].values.tolist()]
    features_func = lambda s: [np.array(s.drop(columns=["sentnum", "ia", "lang", "trialid", "ianum", "trial_sentnum"]).iloc[i])
                                for i in range(len(s))]

    sentences = data.groupby("sentnum").apply(word_func).tolist()

    targets = data.groupby("sentnum").apply(features_func).tolist()

    return sentences, targets


def load_data(filename=None):
    data = pd.read_csv(filename, index_col=0)

    print(data.head(), end="\n\n")

    sentences, targets = create_senteces_from_data(data)

    # split in train, valid, test
    train_sentences, test_sentences, train_targets, test_targets = train_test_split(sentences, targets, shuffle=False, test_size=0.10)
    train_sentences, valid_sentences, valid_targets, valid_targets = train_test_split(train_sentences, train_targets, shuffle=False, test_size=0.15)

    print(f"Train elements : {len(train_sentences)}", end="\n\n")
    print(f"Valid elements : {len(valid_sentences)}", end="\n\n")
    print(f"Test elements : {len(test_sentences)}", end="\n\n")

    return train_sentences, valid_sentences, test_sentences, train_targets, valid_targets, test_targets


def main():
    train_sentences, valid_sentences, test_sentences, train_targets, valid_targets, test_targets = load_data("datasets/cluster_0_dataset.csv")


if __name__ == "__main__":
    main()