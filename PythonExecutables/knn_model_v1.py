#work in progress, one day pred only
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import csvreadwritetemplate as csv

#change if u want
TEST = 0.2
DAYS = 7
PRED_DAYS = 1
K = 50

#train-test split with respect to the consecutive days
#didin't use scikit's built-it because we need to preserve consec days
def training_split(data,test_per):
    cut_index = len(data) - int(len(data)*test_per) - 1
    print(cut_index)
    train = data[:cut_index]
    test = data[cut_index:]
    return train, test

#splits df columns to two (depending on how many days)
def split_col(data,d_cut,col_len):
    cut_index = (len(data.columns)) - (d_cut*col_len)
    new_train = data.iloc[:,:cut_index]
    result = data.iloc[:,cut_index:]
    return new_train, result

#transform the dataset depending on the # of days
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
    t,r = split_col(transform_train_data,d_cut,len(train_data.columns))
    return t,r

input_file = input("Input file location: ")
df = csv.read_csv_file(input_file)

train_data, test_data = training_split(df,TEST)
print(train_data)
print(test_data)

x_train, y_train = transform_data(train_data,DAYS,PRED_DAYS)
print(x_train)
print(y_train)

x_test, y_test = transform_data(test_data,DAYS,PRED_DAYS)
print(x_test)
print(y_test)

#scikit stuff  below here [W.I.P.]

knn_regressor = KNeighborsRegressor(n_neighbors=K)
knn_regressor.fit(x_train, y_train)
y_pred_arr = knn_regressor.predict(x_test)
col_names = [f"{col}" for col in y_test.columns]
y_pred = pd.DataFrame(y_pred_arr, columns=col_names)

print(y_pred)
print(y_test)

y_abs_diff = (y_test.subtract(y_pred)).abs()

print(y_abs_diff)

MAE_list = []
for i in range(len(y_abs_diff.columns)):
    MAE_col = ((y_abs_diff.iloc[:,i]).sum())/(len(y_abs_diff))
    MAE_col = MAE_col.tolist()
    MAE_list.append(MAE_col)

print(MAE_list)
print("\n")

for i in range(len(MAE_list)):
    MAPE = MAE_list[i]/(y_pred.iloc[:,i].sum()/len(y_pred))
    print(1-MAPE)