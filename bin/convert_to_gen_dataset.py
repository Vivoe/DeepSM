"""
Convert Datasets for the Step Placement models into datasets for the
Step Generation model.
"""

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description="Converts a placement dataset into a dataset"
            "for training step generation networks.")

    parser.add_argument('base_dataset', type=str,
            help="The dataset to convert.")
    parser.add_argument('output_name', type=str)
    parser.add_argument('--raw_data', type=str, default=None,
            help="Raw data directory, used for perfect BPM estimation.")

    args = parser.parse_args()

    convert_to_gen_dataset.convert_dataset(args.base_dataset, args.output_name,
            raw_data_name=args.raw_data)

