import argparse

from deepSM import SMDUtils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Splits dataset into train/test splits.")

    parser.add_argument('dataset', type=str)
    parser.add_argument('--test', type=float, default=0.25,
            help="Percent of data in test dataset.")

    args = parser.parse_args()

    SMDUtils.train_test_split_dataset(args.dataset, test_split=args.test)
