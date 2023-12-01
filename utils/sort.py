import os
import pandas as pd

# Temporary code that retreives results, sorts and processes them (calculates mean, reorder columns etc.)

# Initialize an empty DataFrame to store the data
all_data = pd.DataFrame()

# Walk through the directory
for root, dirs, files in os.walk('results'):
    for file in files:
        # Check if the file is 'summary.csv'
        if file == 'summary_adni.csv':
            read_file = os.path.join(root, file)
            print('reading file: ', read_file)
            # Construct the full file path
            file_path = os.path.join(root, file)
            # Read the CSV file into a DataFrame
            data = pd.read_csv(file_path)
            # Append the data to the all_data DataFrame
            all_data = all_data.append(data, ignore_index=True)

# Write the all_data DataFrame to a new CSV file
all_data.to_csv('results/_summary/new_summary.csv', index=False)

# **********************************************************************************************************************

# Read the CSV file into a DataFrame
data = pd.read_csv('results/_summary/new_summary.csv')

# Sort the DataFrame by 'method', 'name', and 'seed'
sorted_data = data.sort_values(by=['algo', 'name', 'seed'])

# Define the new column order (only the useful columns)
column_order = ['algo', 'name', 'seed', 'epochs', 'batch_size', 'train_mae', 'test_mae', 'train_mse', 'test_mse', 'train_reward_rl', 'test_reward_rl']

# Rearrange the columns
sorted_data = sorted_data[column_order]

# Write the rearranged DataFrame to a new CSV file
sorted_data.to_csv('results/_summary/sorted_summary.csv', index=False)

# Now we take a mean for all 5 seeds for each algo and name(fold)
# Group the DataFrame by 'algo' and 'name', and calculate the mean of the numerical columns
grouped_data = sorted_data.groupby(['algo', 'name']).mean().reset_index()

# Round off each value to 3 decimal points
grouped_data = grouped_data.round(3)

# replace the seed column with the number of seeds and rename column to 'num_seeds'. Remove the seed column.
grouped_data = grouped_data.drop(columns=['seed'])
grouped_data['num_seeds'] = 5
# place the num_seeds as the 3rd column
grouped_data = grouped_data[['algo', 'name', 'num_seeds', 'epochs', 'batch_size', 'train_mae', 'test_mae', 'train_mse', 'test_mse', 'train_reward_rl', 'test_reward_rl']]
# Rearrange the columns
sorted_data = sorted_data[column_order]

# Write the grouped DataFrame to a new CSV file
grouped_data.to_csv('results/_summary/final_results.csv', index=False)

# Find the index of the row with the least 'test_mae' for each 'algo'
idx = grouped_data.groupby(['algo'])['test_mae'].idxmin()

# Print out the row with the least 'test_mae' for each 'algo'
best_rows = grouped_data.loc[idx]
print(best_rows)

# Get the names from the best_rows
best_names = best_rows['name'].values

# Filter the sorted_data DataFrame to only include rows where the name is in best_names
filtered_data = sorted_data[sorted_data['name'].isin(best_names)]

# Group the filtered_data by 'algo' and 'name' and find the row with the least 'test_mae' for each group
best_seed_idx = filtered_data.groupby(['algo', 'name'])['test_mae'].idxmin()

# Print out the row with the least 'test_mae' for each group
best_seed_rows = filtered_data.loc[best_seed_idx]
print(best_seed_rows)
