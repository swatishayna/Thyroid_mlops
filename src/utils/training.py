
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import gzip
import pickle
import pickletools
import pandas as pd

# def get_correlation(df):
#     corr_values = abs(df.corr(method='spearman')['Class'])
#     corr_values = corr_values.drop('Class')
#     corr_values = corr_values[corr_values > 0.03]
#     return corr_values


def scaled_df(df, scaler_fileppath, label):
    scaler = MinMaxScaler()

    data = pd.DataFrame(scaler.fit_transform(
        df.drop(columns=['Class'])), columns=df.columns[:-1])
    pickle.dump(scaler, open(scaler_fileppath, 'wb'))
    data[label] = df[label]
    return data

# def smote_resampled(X_train, X_test, y_train, y_test):
#     smote = SMOTE('not majority',random_state = 1)
#     X_train_sm, y_train_sm = smote.fit_resample(X_train,y_train)
#     X_test_sm, y_test_sm = smote.fit_resample(X_test,y_test)
#     return X_train_sm,X_test_sm,y_train_sm,y_test_sm


def under_sampled(df, target_col):
    df_negative = df[df[target_col] == 0]
    df_hypothyroid = df[df[target_col] == 1]
    df_hyperthyroid = df[df[target_col] == 2]
    df_sickeuthyroid = df[df[target_col] == 3]

    df_negative_downsampled = resample(
        df_negative, replace=False, n_samples=300, random_state=1)
    df_hypothyroid_downsampled = resample(
        df_hypothyroid, replace=False, n_samples=300, random_state=1)
    df_hyperthyroid_downsampled = resample(
        df_hyperthyroid, replace=False, n_samples=300, random_state=1)

    df_downsampled = pd.concat(
        [df_negative_downsampled, df_hypothyroid_downsampled, df_hyperthyroid, df_sickeuthyroid])
    return df_downsampled
