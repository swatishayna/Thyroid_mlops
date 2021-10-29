#read data from get_data/source and save the data in data/raw

import os
from get_data import read_params,get_data
import argparse

def load_save(config_path):
    config = read_params(config_path)

    df_allhyper = get_data(config_path)[0]
    df_allhypertest = get_data(config_path)[1]
    df_allhypo= get_data(config_path)[2]
    df_allhypotest= get_data(config_path)[3]
    df_thyroid0387= get_data(config_path)[4]
    df_hypothyroid= get_data(config_path)[5]
    df_sickeuthyroid= get_data(config_path)[6]
    df_anntest= get_data(config_path)[7]
    df_anntrain= get_data(config_path)[8]

    # directory path to store the datasets
    raw_dir_path = config['load_data']['raw_dataset']

    # saving the datsaets to desired location
    df_allhyper.to_csv(os.path.join(raw_dir_path,config['data_source']['allhyper']))
    df_allhypertest.to_csv(os.path.join(raw_dir_path,config['data_source']['allhypertest']))
    df_allhypo.to_csv(os.path.join(raw_dir_path,config['data_source']['allhypo']))
    df_allhypotest.to_csv(os.path.join(raw_dir_path,config['data_source']['allhypotest']))
    df_thyroid0387.to_csv(os.path.join(raw_dir_path,config['data_source']['thyroid0387']))
    df_hypothyroid.to_csv(os.path.join(raw_dir_path,config['data_source']['hypothyroid']))
    df_sickeuthyroid.to_csv(os.path.join(raw_dir_path,config['data_source']['sickeuthyroid']))
    df_anntest.to_csv(os.path.join(raw_dir_path,config['data_source']['anntest']))
    df_anntrain.to_csv(os.path.join(raw_dir_path,config['data_source']['anntrain']))



if __name__ =="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_save(config_path = parsed_args.config)

