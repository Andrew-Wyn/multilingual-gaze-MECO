import numpy as np
import pandas as pd


def split_data(data):
    # get the first 10 trial as train data and the remaining two for valid and test
    train_data = data[np.logical_and(data.trialid != 11, data.trialid != 12)]
    valid_data = data[data.trialid == 11]
    test_data = data[data.trialid == 12]

    print(train_data.head(), end="\n\n")
    print(valid_data.head(), end="\n\n")
    print(test_data.head(), end="\n\n")

    return train_data, valid_data, test_data


def create_senteces_from_data(data):
    word_func = lambda s: [w for w in s["ia"].values.tolist()]
    features_func = lambda s: [np.array(s.drop(columns=["sentnum", "ia", "lang", "trialid", "ianum", "trial_sentnum"]).iloc[i])
                                for i in range(len(s))]

    sentences = data.groupby("sentnum").apply(word_func).tolist()

    targets = data.groupby("sentnum").apply(features_func).tolist()

    return sentences, targets


def load_data(filename=None):
    data = pd.read_csv(filename, index_col=0)

    print(data.head())

    # splid data in train, valid, test
    train_data, valid_data, test_data = split_data(data)

    # transform datasets into a sentences targets sequences
    train_sentences, train_targets = create_senteces_from_data(train_data)
    valid_sentences, valid_targets = create_senteces_from_data(valid_data)
    test_sentences, test_targets = create_senteces_from_data(test_data)

    return train_sentences, valid_sentences, test_sentences, train_targets, valid_targets, test_targets


def main():
    train_sentences, valid_sentences, test_sentences, train_targets, valid_targets, test_targets = load_data("cleaned_data.csv")


if __name__ == "__main__":
    main()