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

def clean_ann(df,column_list):
    df = df.iloc[:,0].str.split(' ', expand=True)
    df = df.drop(columns = [22,23])
    assign_columns_to_df(df,column_list)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    df['sex'] = df['sex'].map({0:'F',1:'M'})
    df['Class'] = df['Class'].map({3:'negative',2:'hypothyroid',1:'hyperthyroid'})

    continuos_attributes = ['age','TSH','T3','TT4','T4U','FTI']
    for attribute in continuos_attributes:
        df[attribute] = df[attribute] * 100

    def fillNewAttributes(row,attribute):
        if row[attribute] > 0:
            return 'y'
        else:
            return 'n'

    df['TSH_measured'] = df.apply(lambda row: fillNewAttributes(row,'TSH'), axis=1)
    df['T3_measured'] = df.apply(lambda row: fillNewAttributes(row,'T3'), axis=1)
    df['TT4_measured'] = df.apply(lambda row: fillNewAttributes(row,'TT4'), axis=1)
    df['T4U_measured'] = df.apply(lambda row: fillNewAttributes(row,'T4U'), axis=1)
    df['FTI_measured'] = df.apply(lambda row: fillNewAttributes(row,'FTI'), axis=1)



    return df