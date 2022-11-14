from dataset import GazeDataset


def main():
    d = GazeDataset(10, None, "datasets/cluster_0_dataset.csv", "try")
    d.read_pipeline()
    print(d.text_inputs["train"][0])
    print(d.targets["train"][0])


if __name__ == "__main__":
    main()