from dataset import GazeDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os


# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR


def main():

    model = "prajjwal1/bert-mini"

    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=CACHE_DIR)

    d = GazeDataset(10, tokenizer, "datasets/cluster_0_dataset.csv", "try")
    d.read_pipeline()


if __name__ == "__main__":
    main()