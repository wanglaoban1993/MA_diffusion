import re
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


current_path = os.getcwd()
print("current path:", current_path)

files = os.listdir(current_path)
print("everything unter current path: ", len(files))

def extract_data_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    epochs = re.findall(r'epoch_(\d+)', content)
    accuracies = re.findall(r'Sudoku accuracy: (\d+\.\d+)%', content)
    
    return list(zip(epochs, accuracies))

def save_to_csv(data, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for epoch, correct in data:
            writer.writerow({'epoch': epoch, 'correct': correct})

# List of input and output file paths
input_files = [
    '1358913_test_ddsm.out',
    '1358914_test_reflectboundaries.out',
    '1358916_test_reflection.out'

]
output_files = [
    'original_400_sab.csv',
    'reflectboundaries_400_sab.csv',
    'reflection_400_sab.csv'
]

# Process each file and save results to CSV
for input_file, output_file in zip(input_files, output_files):
    data = extract_data_from_file(input_file)
    save_to_csv(data, output_file)
print("CSV files created successfully.")

# List of CSV files and corresponding DataFrame names
csv_files = {
    'original_400_sab.csv': 'df_original',
    'reflectboundaries_400_sab.csv': 'df_reflectboundaries',
    'reflection_400_sab.csv': 'df_reflection'
}

# Dictionary to store DataFrames
dataframes = {}

# Load each CSV file into a DataFrame and store it in the dictionary
for csv_file, df_name in csv_files.items():
    dataframes[df_name] = pd.read_csv(csv_file)

# Access the DataFrames using their names
df_original = dataframes['df_original']
df_reflectboundaries = dataframes['df_reflectboundaries']
df_reflection = dataframes['df_reflection']


# Create a new figure
plt.figure(figsize=(10, 5))
# Plot the first dataset with different transparency (alpha) values
sns.lineplot(x='epoch', y='correct', data=df_original, label='DDSM', alpha=0.6)
sns.lineplot(x='epoch', y='correct', data=df_reflectboundaries, label='Reflectboundaries', alpha=0.6)
sns.lineplot(x='epoch', y='correct', data=df_reflection, label='Reflection', alpha=0.6)

end_correct_original= df_original['correct'].iloc[-1]
end_correct_reflectboundaries= data=df_reflectboundaries['correct'].iloc[-1]
end_correct_reflection= data=df_reflection['correct'].iloc[-1]

# Add title and labels
plt.title(f"DDSM End: {end_correct_original}\nReflectboundaries End: {end_correct_reflectboundaries}\nReflection End: {end_correct_reflection}", fontsize=10)
plt.xlabel('Epoch')
plt.ylabel('Correct')

# Show the legend
plt.legend()
# Save the combined plot
plt.savefig('combined_correct_400_sab.png')
# Display the plot
plt.show()
plt.close()