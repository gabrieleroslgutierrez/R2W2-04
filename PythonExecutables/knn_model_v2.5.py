'''
KNN V2
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
import csvreadwritetemplate as csv

### change if u want
TEST = 0.2
DAYS = 7
PRED_DAYS = 1 # dont change for now
K = 20
### change if u want

def rain_data_group(data):
    return

def classify_rainfall(rain_data):
    classify_row = pd.DataFrame(columns=["RAINCLASS"])
    for i in range(len(rain_data)):
        if rain_data[i] <= 0:
            classify_row.loc[i] = 0
        else:
            classify_row.loc[i] = 1
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
df = pd.concat([df, class_row], axis=1)

data, vals = transform_data(df,DAYS,PRED_DAYS)

# Feature Distribution Plot
plt.figure(figsize=(10,5))
sns.histplot(df['RAINFALL'], bins=30, kde=True)
plt.title("Rainfall Data Distribution")
plt.xlabel("Rainfall Amount")
plt.ylabel("Frequency")
plt.show()

# Data Transformation Visualization
plt.figure(figsize=(12,6))
plt.plot(range(len(df["RAINFALL"])), df["RAINFALL"], label="Original Rainfall Data")
plt.xlabel("Time")
plt.ylabel("Rainfall")
plt.title("Rainfall Data Transformation Over Time")
plt.legend()
plt.show()

# Data Preparation
X_train, X_test, y_train, y_test = train_test_split(data, vals, test_size=TEST, random_state=4)

data_rain_drop = data.drop(columns=[f"RAINFALL{i}" for i in range(DAYS)])
X_train_drop = X_train.drop(columns=[f"RAINFALL{i}" for i in range(DAYS)])
y_train_drop = y_train.drop(columns=[f"RAINFALL{i}" for i in range(DAYS,DAYS+PRED_DAYS)])
X_test_drop = X_test.drop(columns=[f"RAINFALL{i}" for i in range(DAYS)])
y_test_drop = y_test.drop(columns=[f"RAINFALL{i}" for i in range(DAYS,DAYS+PRED_DAYS)])

X_train_noclass = X_train.drop(columns=[f"RAINCLASS{i}" for i in range(DAYS)])
y_train_noclass = y_train.drop(columns=[f"RAINCLASS{i}" for i in range(DAYS,DAYS+PRED_DAYS)])
X_test_noclass = X_test.drop(columns=[f"RAINCLASS{i}" for i in range(DAYS)])
y_test_noclass = y_test.drop(columns=[f"RAINCLASS{i}" for i in range(DAYS,DAYS+PRED_DAYS)])

y_train_class = y_train_drop[f"RAINCLASS{DAYS}"]
y_test_class = y_test_drop[f"RAINCLASS{DAYS}"]

# Train K-NN Model
KClass_rain = KNeighborsClassifier(n_neighbors=K)
KClass_rain.fit(X_train_drop,y_train_class)

y_pred_class = KClass_rain.predict(X_test_drop)
y_pred_class = pd.DataFrame(y_pred_class)

y_test_class = y_test_class.reset_index(drop=True)
y_pred_class = y_pred_class.reset_index(drop=True)

# Model Predictions vs. Actual Values
plt.figure(figsize=(10,5))
plt.bar(range(len(y_test_class)), y_test_class, color='blue', alpha=0.5, label='Actual')
plt.bar(range(len(y_pred_class)), y_pred_class.iloc[:,0], color='red', alpha=0.5, label='Predicted')
plt.xlabel("Sample Index")
plt.ylabel("Rainfall Class (0 = No Rain, 1 = Rain)")
plt.title("Actual vs. Predicted Rainfall Classification")
plt.legend()
plt.show()

# Accuracy Calculation
count = (y_test_class == y_pred_class.iloc[:,0]).sum()
accuracy = count / len(y_pred_class)
print(f"Correct Predictions: {count}")
print(f"K-NN Classification Accuracy: {accuracy:.2f}")
