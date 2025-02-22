#work in progress, one day pred only
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor as knn_reg
import csvreadwritetemplate as csv

#change if u want
TEST = 0.2

#train-test split with respect to the consecutive days
#didin't use scikit's built-it because we need to preserve consec days
def training_split(data,test_per):
    cut_index = len(data) - int(len(data)*test_per) - 1
    print(cut_index)
    train = data[:cut_index]
    test = data[cut_index:]
    return train,test

#splits df columns to two
def split_col(data,first_col_size):
    print("hatdog")

def transform_train(train_data,days):
    for i in range(len(train_data)-5):
        print(train_data.ilocp[0,1])
    return 

input_file = input("Input file location: ")
df = csv.read_csv_file(input_file)

train_data, test_data = training_split(df,TEST)
print(train_data)
print(test_data)


