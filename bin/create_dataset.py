"""
Creates/feature engineers datasets ready for training/testing from raw data.
"""

import os
import argparse

from deepSM import SMDUtils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Convert raw dataset into step placement dataset.")

    # Looks for the folder data/{dataset}.
    parser.add_argument('dataset', type=str, help='The raw data to process.')
    # Outputs to datasets/{output_name}.
    parser.add_argument('output_name', type=str)
    parser.add_argument('--drop_diffs', type=str, nargs='+',
            help="Exclude difficulty from processing.")
    parser.add_argument('--test', type=float, default=-1,
            help="Percent of data in test dataset, if splitting dataset.")


    args = parser.parse_args()

    SMDUtils.save_generated_datasets(args.dataset,
            dataset_name=args.output_name,
            test_split=None,
            drop_diffs=args.drop_diffs)

    if args.test >= 0:
        SMDUtils.train_test_split_dataset(args.output_name, args.test)

