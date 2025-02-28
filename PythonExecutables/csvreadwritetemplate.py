import pandas as pd

# Reading a CSV File
def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        print("CSV File Read Successfully!")
        print(data)
        return data
    except Exception as e:
        print(f"Error Reading CSV File: {e}")
        raise Exception("Invalid FILE")
        return None

# Writing to a CSV File
def write_csv_file(data, output_path):
    try:
        data.to_csv(output_path, index=False)  # index=False avoids writing row numbers
        print(f"CSV File Written Successfully to {output_path}")
    except Exception as e:
        print(f"Error Writing CSV File: {e}")

# Example Usage
if __name__ == "__main__":
    # File paths
    input_file = "input.csv"    # Replace with your input file path
    output_file = "output.csv"  # Replace with your desired output file path
    
    # Reading the CSV file
    df = read_csv_file(input_file)
    
    if df is not None:
        # Making changes (optional: e.g., adding a new column)
        df["New_Column"] = "Sample Data"
        
        # Writing the modified DataFrame to a new CSV file
        write_csv_file(df, output_file)
