import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from get_data import read_params

import argparse
import json
import gzip
import pickle
import pickletools


def trainevaluate(config_path):
    config = read_params(config_path)

    # loading data from data/processed
    dirpath = config['data_cleaning_processing']['processed_dataset']
    train_filename = config['traintestsplit']['train_filename']
    train_filepath = os.path.join(dirpath, train_filename)
    test_filename = config['traintestsplit']['test_filename']
    test_filepath = os.path.join(dirpath, test_filename)

    train = pd.read_csv(train_filepath)
    test = pd.read_csv(test_filepath)

    # Note: LIst of important features has been selected
    feature_col = config['train_evaluate']['selected_features']
    target_col = config['base']['target_col']

    random_state = config['base']['random_state']
    class_weight = config['train_evaluate']['class_weight']
    scaler_model_dir = config['traintestsplit']['scaler_model_dir']
    model_filename = config['train_evaluate']['model_filename']
    model_filepath = os.path.join(scaler_model_dir, model_filename)

    # scaling the train_data
    #train_data = scaled_df(train[all_col],scaler_filepath,target_col)
    # data is highly imbalanced we will perform undersampling on train_data
    #train_downsampled = under_sampled(train_data[all_col],target_col)

    # scaling the test_data
    #test_data = scaled_df(test[all_col],scaler_filepath,target_col)
    # data is highly imbalanced we will perform undersampling on test_data
    #test_downsampled = under_sampled(test_data[all_col],target_col)

    # Features
    train_x = train[feature_col]
    test_x = test[feature_col]
    # Label
    train_y = train[target_col]
    test_y = test[target_col]

    # Note: Checked different models and found Random Forest Performing best
    rfc = RandomForestClassifier(
        class_weight=class_weight, random_state=random_state)
    rfc.fit(train_x, train_y)

    # prediction
    y_pred = rfc.predict(test_x)
    precision_score, recall_score, f_score, support = metrics.precision_recall_fscore_support(
        test_y, y_pred, average='macro')

    scores_file = config["reports"]["scores"]

    with open(scores_file, "a") as f:
        scores = {
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f_score": f_score
        }
        json.dump(scores, f, indent=4)

    with gzip.open(model_filepath, "wb") as f:
        pickled = pickle.dumps(rfc)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    trainevaluate(config_path=parsed_args.config)
