#standardizing values (testing)
import csvreadwritetemplate as csv
import pandas as pd

#func to search min value
def search_min(data_list):
    min = 0
    for i in range(len(data_list)):
        val = data_list.iloc[i]
        if val < min: min = val
    print(min)
    return min

#func to search max value
def search_max(data_list):
    max = 0
    for i in range(len(data_list)):
        val = data_list.iloc[i]
        if val > max: max = val
    print(max)
    return max

#func to standardize given min max values
def standardize(data_list,min,max):
    print("test")

#read csv file
df = csv.read_csv_file("../[Original Values Only] Science Garden.csv")

#define empty df
std_df = pd.DataFrame()
print(std_df)

for i in range(len(df.columns)):
    min = search_min(df.iloc[:,i])
    max = search_max(df.iloc[:,i])
    std_col = standardize(df.iloc[:,i],min,max)
    std_df = pd.concat([std_df, std_col], axis=1, join="outer")

print(std_df)