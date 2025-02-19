#standardizing values
import csvreadwritetemplate as csv
import pandas as pd
import sys

#func to search min value
def search_min(data_list):
    min = 0
    for i in range(len(data_list)):
        val = data_list.iloc[i]
        if i == 0: min = val
        if val < min: min = val
    return min

#func to search max value
def search_max(data_list):
    max = 0
    for i in range(len(data_list)):
        val = data_list.iloc[i]
        if i == 0: max = val
        if val > max: max = val
    return max

#func to standardize given min max values (minmax scaling)
def standardize(data_list, min, max):
    diff = abs(max - min)
    std_values = [(abs(val - min) / diff) for val in data_list]
    return pd.DataFrame(std_values, columns=[data_list.name])

#read csv file

input_file = input("Input file location: ")
df = csv.read_csv_file(input_file)

#define empty df
std_df = pd.DataFrame()

'''
go through each column to standardize them
-searches each column min and max values
-standardize them using minmax scaling
-concatenate them to the empty DataFrame (std_df)
'''
try:
    for i in range(len(df.columns)):
        min = search_min(df.iloc[:,i])
        max = search_max(df.iloc[:,i])
        std_col = standardize(df.iloc[:,i],min,max)
        std_df = pd.concat([std_df, std_col], axis=1, join="outer")
except Exception as e:
    print(f"Something went wrong: {e}")
    sys.exit()


try:
    print(f"\n{std_df}")
    output_file = input("Save CSV File to (include filename in path <path>\\<filename>.csv): ")
    std_df.to_csv(output_file)
except Exception as e:
    print(f"Failed to save at the specified location: {e}")
    print("Saving to default directory")
    std_df.to_csv("output.csv")