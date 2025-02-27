import pandas as pd
import csvreadwritetemplate as csv

# Function to split the dataframe into two parts based on the cut_index
def split_col(data, cut_index):
    df1 = data.iloc[:, :cut_index]
    df2 = data.iloc[:, cut_index:]
    return df1, df2

def check_majority(dates,date_x,col):
    majority = False
    major = 0
    count = 0
    print("///")
    for i in range(len(dates)):
        if date_x["MONTH"] == dates.iloc[i,1] and date_x["DAY"] == dates.iloc[i,2]:
            if date_x["YEAR"] != dates.iloc[i,0] and col[i] > 0:
                print(col[i])
                count = count + 1
                major = major + 1
            else: 
                if date_x["YEAR"] != dates.iloc[i,0] and col[i] <= 0:  
                    print(col[i])
                    count = count + 1
    print("///")
    print(major)
    print(count)
    print("///")
    if major/count > 0.5:
        majority = True
    else:
        majority = False
    return majority

def date_average(dates,date_x,col):
    val = 0
    count = 0  
    for i in range(len(dates)):
        if date_x["MONTH"] == dates.iloc[i,1] and date_x["DAY"] == dates.iloc[i,2]:
            if date_x["YEAR"] != dates.iloc[i,0] and col[i] >= 0:
                val = val + col[i]
                print(col[i])   
                count = count + 1
    if count == 0: count = 1
    val = val/count
    return val

def rainfall_date_average(dates,date_x,col):
    val = 0
    if check_majority(dates,date_x,col) == False:
        val = 0
    else:
        count = 0  
        for i in range(len(dates)):
            if date_x["MONTH"] == dates.iloc[i,1] and date_x["DAY"] == dates.iloc[i,2]:
                if date_x["YEAR"] != dates.iloc[i,0] and col[i] > 0:
                    val = val + col[i]
                    print(col[i])   
                    count = count + 1
        if count == 0: count = 1
        val = val/count
    print("---")
    print(val)
    print("---")
    return val

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
print(dates)

# Initialize empty dataframes for average and line interpolation results
ave = pd.DataFrame()

# Iterate over each column in the data
for col in data:
    dat = data[col]
    print(dat.name)
    for i in range(len(dat)):
        val = dat.iloc[i]
        print(val)
        if val < 0:
            if dat.name == "RAINFALL":
                print("yes2")
                dv = rainfall_date_average(dates,dates.iloc[i],dat)
                dat[i] = dv
            else:
                print("yes")
                dv = date_average(dates,dates.iloc[i],dat)
                dat[i] = dv
    ave = pd.concat([ave, dat], axis=1, ignore_index=True)

print(ave)
ave.to_csv("testing.csv", index=False)
    