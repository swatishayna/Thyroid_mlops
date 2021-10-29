import pandas as pd


def assign_columns_to_df(df, column_list):
    df.columns = column_list

def removenumber(column):
    """removing the numerical value 
    from the end of the Target column"""

    for i in column:
        i = i.split('.')[0]
        return i

def cleanclasscolumn(df, given_target_values,target_true):
    df_negative = df[df.Class=='negative']
    df_cleaning = df[df.Class.isin(given_target_values)]
    df_cleaning.replace(given_target_values,target_true, inplace =True)
    df=pd.concat([df_negative,df_cleaning],ignore_index = True)
    return df


def clean_thyroid0387(df_thyroid0387):
    # Target Column Analysis
    # cleaning Target Column
    Class = []
    for records in df_thyroid0387.Target:
        Class.append(str(records).split("[")[0])
    # Adding the as Class Column into the dataframe
    df_thyroid0387['Class'] = pd.Series(Class)
    # Set A,B,C,D to hyperthyroid
    # Set E,F,G,H to hypothyroid
    # Rest all classes will be set to negative
    df_thyroid0387['Class'].replace(['A','B','C','D'],"hyperthyroid",inplace = True)
    df_thyroid0387['Class'].replace(['E','F','G','H'],"hypothyroid",inplace = True)
    for classes in df_thyroid0387['Class'].value_counts().index:
        if classes != "hyperthyroid" and classes !="hypothyroid":
            df_thyroid0387['Class'].replace(classes,"negative",inplace = True)
    
    
    df_thyroid0387.drop(columns = ["Target"], inplace =True)
    df_thyroid0387.drop_duplicates(inplace =True)


    return df_thyroid0387


def clean_hypothyroid_sickeuthyroid(df):
    # we have 'Sex' and 'Age'column which we have to make 'sex' and 'age'
    # df_hypothyroid.Unnamed is our target/Class column we have to rename it
    df = df.rename(columns = {
        df.columns[0]: "Class",
        df.columns[1] :  "age",
        df.columns[2] : "sex"
                                    })


    return df