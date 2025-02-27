'''
KNN V2
'''
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import csvreadwritetemplate as csv

### change if u want
TEST = 0.2
DAYS = 7
PRED_DAYS = 1 # dont change for now
K = 20
### change if u want

def classify_rainfall(rain_data):
    classify_row = pd.DataFrame(columns=["RAINCLASS"])
    for i in range(len(rain_data)):
        if rain_data[i] <= 0:
            classify_row.loc[i] = 0
        else:
            classify_row.loc[i] = 1
    print(classify_row)
    return classify_row

def split_col(data,d_cut,col_len):
    cut_index = (len(data.columns)) - (d_cut*col_len)
    new_train = data.iloc[:,:cut_index]
    result = data.iloc[:,cut_index:]
    return new_train, result

def transform_data(train_data,days,d_cut):
    transform_train_data = pd.DataFrame()
    for i in range(len(train_data) - days):
        transform_row = pd.DataFrame()
        for j in range(days + d_cut):
            row = train_data.iloc[i + j, :]
            row.index = [f"{col}{j}" for col in train_data.columns]
            row = row.to_frame().T.reset_index(drop=True)
            if transform_row.empty:
                transform_row = row
            else:
                transform_row = pd.concat([transform_row, row], axis=1)
        transform_train_data = pd.concat([transform_train_data, transform_row], axis=0, ignore_index=True)
    # t,r = split_col(transform_train_data,d_cut,len(train_data.columns))
    return transform_train_data

input_file = input("Input file location: ")
df = csv.read_csv_file(input_file)

class_row = classify_rainfall(df["RAINFALL"])
df = pd.concat([df, class_row], axis=1)

data_rain_drop

