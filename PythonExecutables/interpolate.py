import pandas as pd
import csvreadwritetemplate as csv

# Function to split the dataframe into two parts based on the cut_index
def split_col(data, cut_index):
    df1 = data.iloc[:, :cut_index]
    df2 = data.iloc[:, cut_index:]
    return df1, df2

# Placeholder function for date averaging
def date_average(dates, date_x, col):
    return

# Placeholder function for linear interpolation
def line_interpolate(a_index, b_index, a_val, b_val, gap):
    return

# Prompt user for input file location and read the CSV file
input_file = input("Input file location: ")
df = csv.read_csv_file(input_file)

# Split the dataframe into dates and data
dates, data = split_col(df, 3)

# Initialize empty dataframes for storing results
data_average = pd.DataFrame()
line_interpolation = pd.DataFrame()

# Print the data for debugging purposes
print(data)

# Initialize empty dataframes for average and line interpolation results
ave = pd.DataFrame()
line = pd.DataFrame()

# Iterate over each column in the data
for col in data:
    dat = data[col]
    temp_ave = pd.DataFrame(columns=[f"{col}"])
    temp_line = pd.DataFrame(columns=[f"{col}"])
    # Iterate over each value in the column
    for i in range(len(dat)):
        val = dat.iloc[i]
        # If the value is negative, perform date averaging
        if val < 0:
            dv = date_average(dates, dates.iloc[i], dat)
            temp_ave.loc[len(temp_ave)] = dv
    # Concatenate the temporary average dataframe with the main average dataframe
    ave = pd.concat([ave, temp_ave], axis=1, join="outer")