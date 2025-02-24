import pandas as pd

def replace_negative_one(csv_file, output_file):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Ensure numerical values are correctly processed
        df = df.applymap(lambda x: 0 if x == -1 else x)
        
        # Save the modified data to a new CSV file
        df.to_csv(output_file, index=False)
        
        print(f"Modified CSV saved as {output_file}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
input_csv = "[Missing Values Dropped] Science Garden.csv"   # Replace with your actual input file
output_csv = "output.csv"  # Replace with your desired output file
replace_negative_one(input_csv, output_csv)