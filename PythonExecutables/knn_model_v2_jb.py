'''
KNN V2
'''
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import csvreadwritetemplate as csv

### change if u want
TEST = 0.25
DAYS = 45
PRED_DAYS = 1 # dont change for now
K = 50
### change if u want

def rain_data_group(data,ydata):
    data = data.reset_index(drop=True)
    ydata = ydata.reset_index(drop=True)
    out_x = pd.DataFrame(columns=data.columns)
    out_y = pd.DataFrame(columns=ydata.columns)
    for i in range(len(data)):
        if ydata.loc[i,f"RAINCLASS{DAYS}"] == 1:
            x = (pd.DataFrame(data.iloc[i])).T
            y = (pd.DataFrame(ydata.iloc[i])).T
            out_x = pd.concat([out_x, x], ignore_index=True)
            out_y = pd.concat([out_y, y], ignore_index=True)

    print(out_x)
    out_y = out_y.drop(columns=[f"RAINCLASS{DAYS}"])
    print(out_y)
    return out_x, out_y

def no_rain_data_group(data,ydata):
    data = data.reset_index(drop=True)
    ydata = ydata.reset_index(drop=True)
    out_x = pd.DataFrame(columns=data.columns)
    out_y = pd.DataFrame(columns=ydata.columns)
    for i in range(len(data)):
        if ydata.loc[i,f"RAINCLASS{DAYS}"] == 0:
            x = (pd.DataFrame(data.iloc[i])).T
            y = (pd.DataFrame(ydata.iloc[i])).T
            out_x = pd.concat([out_x, x], ignore_index=True)
            out_y = pd.concat([out_y, y], ignore_index=True)
    print(out_x)
    out_y = out_y.drop(columns=[f"RAINCLASS{DAYS}",f"RAINFALL{DAYS}"])
    print(out_y)
    return out_x, out_y

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
    t,r = split_col(transform_train_data,d_cut,len(train_data.columns))
    return t,r

#get file
input_file = input("Input file location: ")
df = csv.read_csv_file(input_file)

class_row = classify_rainfall(df["RAINFALL"])
df_2 = pd.concat([df, class_row], axis=1)

data, vals = transform_data(df_2,DAYS,PRED_DAYS)
print(data)
print(vals)

#data stuff
X_train, X_test, y_train, y_test = train_test_split(data, vals, test_size=TEST, random_state=4)

X_train = pd.DataFrame(X_train)
print(X_train)
y_train = pd.DataFrame(y_train)
print(y_train)
X_test = pd.DataFrame(X_test)
print(X_test)
y_test = pd.DataFrame(y_test)
print(y_test)

data_rain_drop = data.drop(columns=[f"RAINFALL{i}" for i in range(DAYS)])
print(data_rain_drop)

X_train_drop = X_train.drop(columns=[f"RAINFALL{i}" for i in range(DAYS)])
print(X_train_drop)
y_train_drop = y_train.drop(columns=[f"RAINFALL{i}" for i in range(DAYS,DAYS+PRED_DAYS)])
print(y_train_drop)
X_test_drop = X_test.drop(columns=[f"RAINFALL{i}" for i in range(DAYS)])
print(X_test_drop)
y_test_drop = y_test.drop(columns=[f"RAINFALL{i}" for i in range(DAYS,DAYS+PRED_DAYS)])
print(y_test_drop)

X_train_noclass = X_train.drop(columns=[f"RAINCLASS{i}" for i in range(DAYS)])
print(X_train_noclass)
y_train_noclass = y_train.drop(columns=[f"RAINCLASS{i}" for i in range(DAYS,DAYS+PRED_DAYS)])
print(y_train_noclass)
X_test_noclass = X_test.drop(columns=[f"RAINCLASS{i}" for i in range(DAYS)])
print(X_test_noclass)
y_test_noclass = y_test.drop(columns=[f"RAINCLASS{i}" for i in range(DAYS,DAYS+PRED_DAYS)])
print(y_test_noclass)

y_train_class = y_train_drop[f"RAINCLASS{DAYS}"]
print(y_train_class)
y_test_class = y_test_drop[f"RAINCLASS{DAYS}"]
print(y_test_class)

X_train_raingroup, y_train_raingroup = rain_data_group(X_train_noclass,y_train)
X_train_noraingroup, y_train_noraingroup = no_rain_data_group(X_train_noclass,y_train)
###

KClass_rain = KNeighborsClassifier(n_neighbors=K)
KClass_rain.fit(X_train_drop,y_train_class)

KRegressor_raingroup = KNeighborsRegressor(n_neighbors=K)
KRegressor_raingroup.fit(X_train_raingroup,y_train_raingroup)

KRegressor_noraingroup = KNeighborsRegressor(n_neighbors=K)
KRegressor_noraingroup.fit(X_train_noraingroup,y_train_noraingroup)

predictions = pd.DataFrame(columns=df.columns)
for i in range(len(X_test_drop)):
    y_pred_class = KClass_rain.predict((pd.DataFrame(X_test_drop.iloc[i])).T)
    print(y_pred_class.tolist())
    if y_pred_class == 0:
        #print("regression, rain = 0")
        y_pred = KRegressor_noraingroup.predict((pd.DataFrame(X_test_noclass.iloc[i])).T)
        new_value = np.array([[0.0]])  # New value to add
        y_pred = np.insert(y_pred, 0, new_value, axis=1) 
        pred = pd.DataFrame(y_pred,columns=df.columns)
        predictions = pd.concat([predictions, pred], axis=0, ignore_index=True)
        print(pred)
    else:
        if y_pred_class == 1:
            #print("regression with rain")
            y_pred = KRegressor_raingroup.predict((pd.DataFrame(X_test_noclass.iloc[i])).T)
            pred = pd.DataFrame(y_pred,columns=df.columns)
            predictions = pd.concat([predictions, pred], axis=0, ignore_index=True)
            print(pred)

print(predictions)
y_test_noclass = y_test_noclass.rename(columns={f"RAINFALL{DAYS}": "RAINFALL",f"TMAX{DAYS}": "TMAX",f"TMIN{DAYS}": "TMIN",f"RH{DAYS}": "RH",f"WIND_SPEED{DAYS}": "WIND_SPEED",f"WIND_DIRECTION{DAYS}": "WIND_DIRECTION",f"BAROMETRIC_AIR_PRESSURE{DAYS}": "BAROMETRIC_AIR_PRESSURE"})
y_test_noclass = y_test_noclass.reset_index(drop=True)
print(y_test_noclass)

predictions.to_csv("1.csv", index=False)
y_test_noclass.to_csv("2.csv", index=False)

y_abs_diff = (y_test_noclass.subtract(predictions)).abs()

print(y_abs_diff)

MAE_list = []
for i in range(len(y_abs_diff.columns)):
    MAE_col = ((y_abs_diff.iloc[:,i]).sum())/(len(y_abs_diff))
    MAE_col = MAE_col.tolist()
    MAE_list.append(MAE_col)

print(MAE_list)
print("\n")

for i in range(len(MAE_list)):
    MAPE = MAE_list[i]/(predictions.iloc[:,i].sum()/len(predictions))
    print(MAPE)

