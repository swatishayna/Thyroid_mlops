# split the processed dataset into train and test
# save the train and test to data/processed

import os
from get_data import read_params
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.training import scaled_df, under_sampled


def split_traintest_save(config_path):
    config = read_params(config_path)

    # path to processed diectory to fetch processed dataset and store train and test dataset
    dirpath = config['data_cleaning_processing']['processed_dataset']

    processed_data_filename = config['data_cleaning_processing']['processed_filename']
    processed_data_filepath = os.path.join(dirpath, processed_data_filename)

    train_filename = config['traintestsplit']['train_filename']
    train_filepath = os.path.join(dirpath, train_filename)
    test_filename = config['traintestsplit']['test_filename']
    test_filepath = os.path.join(dirpath, test_filename)

    all_col = config['traintestsplit']['all_data']
    scaler_model_dir = config['traintestsplit']['scaler_model_dir']
    scaler_filename = config['traintestsplit']['scaler_filename']
    scaler_filepath = os.path.join(scaler_model_dir,    scaler_filename)
    target_col = config['base']['target_col']

    random_state = config['base']['random_state']
    split_ratio = config['traintestsplit']['test_size']

    # reading the dataset
    df = pd.read_csv(processed_data_filepath)

    # scaling the dataset
    df_scaled = scaled_df(df[all_col], scaler_filepath, target_col)
    # data is highly imbalanced we will perform undersampling on train_data
    train_downsampled = under_sampled(df_scaled[all_col], target_col)

    # splitting the data set
    train, test = train_test_split(train_downsampled,
                                   test_size=split_ratio,
                                   random_state=random_state
                                   )

    train.to_csv(train_filepath, sep=',', index=False, encoding='utf-8')
    test.to_csv(test_filepath, sep=',', index=False, encoding='utf-8')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_traintest_save(config_path=parsed_args.config)
