import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import pad_sequences
from gaze.utils import LOGGER


class GazeDataset():
    def __init__(self, cf, tokenizer, filename, task):
        self.tokenizer = tokenizer  # tokenizer for the BERT model
        self.filename = filename
        self.task = task

        self.modes = ["train", "valid", "test"]

        self.text_inputs = []
        self.targets = []
        self.masks = []  # split key padding attention masks for the BERT model
        self.maps = []  # split mappings between tokens and original words
        self.numpy = []  # split numpy arrays, ready for the model

        self.feature_max = cf.feature_max  # gaze features will be standardized between 0 and self.feature_max

    def tokenize_and_map(self, sentence):
        """
        Tokenizes a sentence, and returns the tokens and a list of starting indices of the original words.
        """
        tokens = []
        map = []

        for w in sentence:
            map.append(len(tokens))
            tokens.extend(self.tokenizer.tokenize(w) if self.tokenizer.tokenize(w) else [self.tokenizer.unk_token])

        return tokens, map

    def tokenize_from_words(self):
        """
        Tokenizes the sentences in the dataset with the pre-trained tokenizer, storing the start index of each word.
        """
        LOGGER.info(f"Tokenizing sentences for task {self.task}")
        tokenized = []
        maps = []

        for s in self.text_inputs:
            tokens, map = self.tokenize_and_map(s)

            tokenized.append(tokens)
            maps.append(map)
            #print(tokens)
        print("max tokenized seq len: ", max(len(l) for l in tokenized))

        self.text_inputs = tokenized
        self.maps = maps

    def calc_input_ids(self):
        """
        Converts tokens to ids for the BERT model.
        """
        LOGGER.info(f"Calculating input ids for task {self.task}")
        ids = [self.tokenizer.prepare_for_model(self.tokenizer.convert_tokens_to_ids(s))["input_ids"]
                for s in self.text_inputs]
        self.text_inputs = pad_sequences(ids, value=self.tokenizer.pad_token_id, padding="post")

    def calc_attn_masks(self):
        """
        Calculates key paddding attention masks for the BERT model.
        """
        LOGGER.info(f"Calculating attention masks for task {self.task}")
        self.masks = [[j != self.tokenizer.pad_token_id for j in i] for i in self.text_inputs]

    def read_pipeline(self):
        self.load_data()

        self.d_out = len(self.targets[0][0])  # number of gaze features
        self.target_pad = -1

        # self.standardize()
        self.tokenize_from_words()
        self.pad_targets()
        self.calc_input_ids()
        self.calc_attn_masks()
        self.calc_numpy()

    def _create_senteces_from_data(self, data):
        word_func = lambda s: [w for w in s["ia"].values.tolist()]
        features_func = lambda s: [np.array(s.drop(columns=["sentnum", "ia", "lang", "trialid", "ianum", "trial_sentnum"]).iloc[i])
                                    for i in range(len(s))]

        sentences = data.groupby("sentnum").apply(word_func).tolist()

        targets = data.groupby("sentnum").apply(features_func).tolist()

        return sentences, targets

    def load_data(self):
        LOGGER.info(f"Loading data for task {self.task}")
        
        dataset = pd.read_csv(self.filename, index_col=0)

        sentences, targets = self._create_senteces_from_data(dataset)

        self.text_inputs = sentences
        self.targets = targets

        LOGGER.info(f"Lenght of data : {len(self.text_inputs)}")

        # split in train, valid, test
        # train_sentences, test_sentences, train_targets, test_targets = train_test_split(sentences, targets, shuffle=False, test_size=0.10)
        # train_sentences, valid_sentences, valid_targets, valid_targets = train_test_split(train_sentences, train_targets, shuffle=False, test_size=0.15)

        # self.text_inputs["train"] = train_sentences
        # self.targets["train"] = train_targets
        #LOGGER.info(f"Lenght of Train data : {len(self.text_inputs['train'])}")

        # self.text_inputs["valid"] = valid_sentences
        # self.targets["valid"] = valid_targets
        # LOGGER.info(f"Lenght of Valid data : {len(self.text_inputs['valid'])}")
        
        # self.text_inputs["test"] = test_sentences
        # self.targets["test"] = test_targets
        # LOGGER.info(f"Lenght of Test data : {len(self.text_inputs['test'])}")

        # # check for duplicate sentence in train and test set
        # dups = []
        # for i, s in enumerate(self.text_inputs["train"]):
        #     if s in self.text_inputs["test"]:
        #         LOGGER.warning("Duplicate in test set....")
        #         dups.append(i)

        # # remove duplicated from training data
        # print(len(dups))
        # for d in sorted(dups, reverse=True):
        #     del self.text_inputs["train"][d]
        #     del self.targets["train"][d]
        # LOGGER.info(f"Lenght of Train data after removed duplicates : {len(self.text_inputs['train'])}")

    # TODO: move to 10-fold-cv
    def standardize(self):
        """
        Standardizes the features between 0 and self.feature_max.
        """
        LOGGER.info(f"Standardizing target data for task {self.task}")
        features = self.targets["train"]
        scaler = MinMaxScaler(feature_range=[0, self.feature_max])
        flat_features = [j for i in features for j in i]
        scaler.fit(flat_features)

        self.targets["train"] = [list(scaler.transform(i)) for i in features]
        self.targets["valid"] = [list(scaler.transform(i)) for i in self.targets["valid"]]
        self.targets["test"] = [list(scaler.transform(i)) for i in self.targets["test"]]

        # filen = os.path.join("scaled-test-"+self.task+".csv")

        #print(filen)

        # flat_preds = [j for i in self.targets["test"] for j in i]

        # preds_pd = pd.DataFrame(flat_preds, columns=["n_fix", "first_fix_dur", "first_pass_dur",
        #                                              "total_fix_dur", "mean_fix_dur", "fix_prob",
        #                                              "n_refix", "reread_prob"])
        # preds_pd.to_csv(filen)

        # print("saved.")

    def pad_targets(self):
        """
        Adds the pad tokens in the positions of the [CLS] and [SEP] tokens, adds the pad
        tokens in the positions of the subtokens, and pads the targets with the pad token.
        """
        LOGGER.info(f"Padding targets for task {self.task}")
        targets = [np.full((len(i), self.d_out), self.target_pad) for i in self.text_inputs]
        for k, (i, j) in enumerate(zip(self.targets, self.maps)):
            targets[k][j, :] = i

        target_pad_vector = np.full((1, self.d_out), self.target_pad)
        targets = [np.concatenate((target_pad_vector, i, target_pad_vector)) for i in targets]

        self.targets = pad_sequences(targets, value=self.target_pad, padding="post")

    def calc_numpy(self):
        LOGGER.info(f"Calculating numpy arrays for task {self.task}")
        self.text_inputs = np.asarray(self.text_inputs, dtype=np.int64)
        self.masks = np.asarray(self.masks, dtype=np.float32)
        self.targets = np.asarray(self.targets, dtype=np.float32)

        # self.numpy = list(zip(input_numpy, target_numpy, mask_numpy))