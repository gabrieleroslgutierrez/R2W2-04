import pandas as pd

def replace_negative_one(csv_file, output_file):
    df = pd.read_csv(csv_file)
    df = df.map(lambda x: 0 if x == -1 else x)
    df.to_csv(output_file, index=False)
    print(f"Modified CSV saved as {output_file}")

input_csv = "[Missing Values Dropped] Science Garden.csv"   
output_csv = "output.csv"  
replace_negative_one(input_csv, output_csv)
