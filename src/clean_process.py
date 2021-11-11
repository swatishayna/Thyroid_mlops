# load saved data from raw
# clean and process the data
# concat process data into one file
# concatenated file will be saved in processed folder


import os
from get_data import read_params
import argparse
import pandas as pd
import numpy as np
from utils.data_cleaning import assign_columns_to_df, removenumber, cleanclasscolumn, clean_thyroid0387, clean_hypothyroid_sickeuthyroid, clean_ann, dropna_thresh, encode


def clean_process(config_path):
    config = read_params(config_path)

    # path to stored raw dataset
    raw_dir_path = config['load_data']['raw_dataset']

    # reading the raw datasets

    df_allhyper = pd.read_csv(os.path.join(
        raw_dir_path, config['data_source']['allhyper']), delimiter=',')
    df_allhypertest = pd.read_csv(os.path.join(
        raw_dir_path, config['data_source']['allhypertest']), delimiter=',')
    df_allhypo = pd.read_csv(os.path.join(
        raw_dir_path, config['data_source']['allhypo']), delimiter=',')
    df_allhypotest = pd.read_csv(os.path.join(
        raw_dir_path, config['data_source']['allhypotest']), delimiter=',')
    df_thyroid0387 = pd.read_csv(os.path.join(
        raw_dir_path, config['data_source']['thyroid0387']), delimiter=',')
    df_hypothyroid = pd.read_csv(os.path.join(
        raw_dir_path, config['data_source']['hypothyroid']), delimiter=',')
    df_sickeuthyroid = pd.read_csv(os.path.join(
        raw_dir_path, config['data_source']['sickeuthyroid']), delimiter=',')
    df_anntest = pd.read_csv(os.path.join(
        raw_dir_path, config['data_source']['anntest']), delimiter=',')
    df_anntrain = pd.read_csv(os.path.join(
        raw_dir_path, config['data_source']['anntrain']), delimiter=',')

    # datasets are not aasigned with columns
    column_hyper_hypo_thy0387 = config['data_cleaning_processing']['column_hyper_hypo_thy0387']
    column_hypothy_sick = config['data_cleaning_processing']['column_hypothy_sick']
    columns_ann = config['data_cleaning_processing']['columns_ann']

    assign_columns_to_df(df_allhyper, column_hyper_hypo_thy0387)
    assign_columns_to_df(df_allhypertest, column_hyper_hypo_thy0387)
    assign_columns_to_df(df_allhypo, column_hyper_hypo_thy0387)
    assign_columns_to_df(df_allhypotest, column_hyper_hypo_thy0387)
    assign_columns_to_df(df_thyroid0387, column_hyper_hypo_thy0387)
    assign_columns_to_df(df_hypothyroid, column_hypothy_sick)
    assign_columns_to_df(df_sickeuthyroid, column_hypothy_sick)

    # removing the numerical value from the end of the Target column
    df_allhyper['Class'] = df_allhyper[['Target']].apply(removenumber, axis=1)
    df_allhypertest['Class'] = df_allhypertest[[
        'Target']].apply(removenumber, axis=1)
    df_allhypo['Class'] = df_allhypo[['Target']].apply(removenumber, axis=1)
    df_allhypotest['Class'] = df_allhypotest[[
        'Target']].apply(removenumber, axis=1)

    df_allhyper = cleanclasscolumn(df_allhyper, [
                                   'hyperthyroid', 'T3 toxic', 'goitre', 'secondary toxic'], 'hyperthyroid')
    df_allhypertest = cleanclasscolumn(df_allhypertest, [
                                       'hyperthyroid', 'T3 toxic', 'goitre', 'secondary toxic'], 'hyperthyroid')
    df_allhypo = cleanclasscolumn(df_allhypo, [
                                  'compensated hypothyroid', 'primary hypothyroid', 'secondary hypothyroid'], 'hypothyroid')
    df_allhypotest = cleanclasscolumn(df_allhypotest, [
                                      'compensated hypothyroid', 'primary hypothyroid', 'secondary hypothyroid'], 'hypothyroid')

    # concatenating df_allhyper,df_allhypertest,df_allhypo,df_allhypotest
    df_hh = pd.concat([df_allhyper, df_allhypertest,
                      df_allhypo, df_allhypotest], ignore_index=True)

    # dropping the duplicate columns
    df_hh.drop_duplicates(inplace=True)
    # Dropping target column
    df_hh.drop(columns=['Target'], inplace=True)

    # cleaning df_thyroid0387
    df_thyroid0387 = clean_thyroid0387(df_thyroid0387)

    # cleaning df_hypothyroid and df_sickeuthyroid
    df_hypothyroid = clean_hypothyroid_sickeuthyroid(df_hypothyroid)
    df_sickeuthyroid = clean_hypothyroid_sickeuthyroid(df_sickeuthyroid)

    # cleaning df_anntest and ann_test
    df_anntest = clean_ann(df_anntest, columns_ann)
    df_anntrain = clean_ann(df_anntrain, columns_ann)

    # concatenating ann train and test
    ann = pd.concat([df_anntrain, df_anntest], ignore_index=True)

    # concatenating df_hh,df_thyroid0387,df_hypothyroid,df_sickeuthyroid,ann
    df = pd.concat([df_hh, df_thyroid0387, df_hypothyroid,
                   df_sickeuthyroid, ann], ignore_index=True)

    # there are "?" entries which denotes missingvalues or nan values replacing them
    df = df.replace({"?": np.NAN})

    # ## Dropping the rows with more than 10 nan values and columns with more than 7000 nans
    df = dropna_thresh(df)

    # Encoding into numerical values
    df = encode(df)

    # the data has very high amount of nan values interploating them
    processed_data = df.interpolate(method='spline', order=3)
    processed_data.dropna(inplace=True)
    print(processed_data.shape)

    # saving the cleaned and processed dataset to the desired location
    processed_data_dirpath = config['data_cleaning_processing']['processed_dataset']
    processed_data_filename = config['data_cleaning_processing']['processed_filename']
    processed_data.to_csv(os.path.join(
        processed_data_dirpath, processed_data_filename), index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    clean_process(config_path=parsed_args.config)
