#work in progress, one day pred only
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
import csvreadwritetemplate as csv

#change if u want
TEST = 0.2
DAYS = 7
PRED_DAYS = 1
K = 20
#train-test split with respect to the consecutive days
#didin't use scikit's built-it because we need to preserve consec days

def rainfall_data(df):
    removed = df[["RAINFALL"]]
    removed.to_csv("rainfall.csv", index = False)
    return removed

def classify_rainfall(df):
    df["CLASSIFIED"] = np.where(df["RAINFALL"]==0,0,1)
    popped = df.pop("RAINFALL")
    return df, popped
    

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
    print("transformed data")
    print(transform_train_data)
    print("-----------------------------------------------")
    return t,r

input_file = input("Input file location: ")
df = csv.read_csv_file(input_file)
unclass_df = rainfall_data(df)
rainfallClass_df, rainfall_df= classify_rainfall(unclass_df)

print("this is rainfall data")
print("--------------------------------------------")
print(rainfallClass_df)

train_rainfall, test_rainfall = training_split(rainfallClass_df,TEST)
print("train, test data")
print(train_rainfall)
print(test_rainfall)
print("-----------------------------------------------")

x_train, y_train = transform_data(train_rainfall,DAYS,PRED_DAYS)
print("x-train")
print(x_train)
print("y-train")
print(y_train)
print("-----------------------------------------------")

x_test, y_test = transform_data(test_rainfall,DAYS,PRED_DAYS)
print("x-test, y-test")
print(x_test)
print(y_test)
print("-----------------------------------------------")

#scikit stuff  below here [W.I.P.]

knn_classifier = KNeighborsClassifier(n_neighbors=K)
knn_classifier.fit(x_train, y_train)
y_pred_arr = knn_classifier.predict(x_test)
col_names = ["Predictions"]
y_pred = pd.DataFrame(y_pred_arr, columns=col_names)

print("y-pred, y-test")
print(y_pred)
print(y_test)
print("-----------------------------------------------")

