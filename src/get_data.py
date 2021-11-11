# read params--> process--> copy into data/raw


import os
import yaml
import pandas as pd
import argparse


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path):
    config = read_params(config_path)

    data_path = config['data_source']['dir']

    # dataframes
    df_allhyper = pd.read_csv(os.path.join(
        data_path, config['data_source']['allhyper']))
    df_allhypertest = pd.read_csv(os.path.join(
        data_path, config['data_source']['allhypertest']))
    df_allhypo = pd.read_csv(os.path.join(
        data_path, config['data_source']['allhypo']))
    df_allhypotest = pd.read_csv(os.path.join(
        data_path, config['data_source']['allhypotest']))
    df_thyroid0387 = pd.read_csv(os.path.join(
        data_path, config['data_source']['thyroid0387']))
    df_hypothyroid = pd.read_csv(os.path.join(
        data_path, config['data_source']['hypothyroid']))
    df_sickeuthyroid = pd.read_csv(os.path.join(
        data_path, config['data_source']['sickeuthyroid']))
    df_anntest = pd.read_csv(os.path.join(
        data_path, config['data_source']['anntest']))
    df_anntrain = pd.read_csv(os.path.join(
        data_path, config['data_source']['anntrain']))

    return df_allhyper, df_allhypertest, df_allhypo, df_allhypotest, df_thyroid0387, df_hypothyroid, df_sickeuthyroid, df_anntest, df_anntrain


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    get_data(config_path=parsed_args.config)
