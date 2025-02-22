import pandas as pd
import numpy as np
import sklearn as sk
import csvreadwritetemplate as csv

input_file = input("Input file location: ")
df = csv.read_csv_file(input_file)

print(df)



