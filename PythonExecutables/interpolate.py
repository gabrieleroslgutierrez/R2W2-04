import pandas as pd
import csvreadwritetemplate as csv

def split_col(data,cut_index):
    df1 = data.iloc[:,:cut_index]
    df2 = data.iloc[:,cut_index:]
    return df1, df2

def date_average(dates,date_x,col):
    
    return

def line_interpolate(a_index,b_index,a_val,b_val,gap):
    return

###
input_file = input("Input file location: ")
df = csv.read_csv_file(input_file)

dates, data = split_col(df,3)

data_average = pd.DataFrame()
line_interpolation = pd.DataFrame()

print(data)

ave = pd.DataFrame()
line = pd.DataFrame()

for col in data:
    dat = data[col]
    temp_ave = pd.DataFrame(columns=[f"{col}"])
    temp_line = pd.DataFrame(columns=[f"{col}"])
    for i in range(len(dat)):
        val = dat.iloc[i]
        if val < 0:
            dv = data_average(dates,date.iloc[i],dat)
            temp_ave.loc[len(temp_ave)] = dv
    ave = pd.concat([ave, temp_ave], axis=1, join="outer")