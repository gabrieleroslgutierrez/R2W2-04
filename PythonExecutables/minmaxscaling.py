#standardizing values (testing) (work in progress)
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

#func to standardize given min max values (minmax scaling)
def standardize(data_list, min, max):
    diff = abs(max - min)
    std_values = [(abs(val - min) / diff) for val in data_list]
    return pd.DataFrame(std_values, columns=[data_list.name])

#read csv file
df = csv.read_csv_file("..\\[Original Values Only] Science Garden.csv")

#define empty df
std_df = pd.DataFrame()
print(std_df)

'''
go through each column to standardize them
-searches each column min and max values
-standardize them using minmax scaling
-concatenate them to the empty DataFrame (std_df)
'''
for i in range(len(df.columns)):
    min = search_min(df.iloc[:,i])
    max = search_max(df.iloc[:,i])
    std_col = standardize(df.iloc[:,i],min,max)
    std_df = pd.concat([std_df, std_col], axis=1, join="outer")

print(std_df)
std_df.to_csv("..\\[Standardized] Science Garden.csv")