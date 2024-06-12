import os
import pandas as pd

def post_process(filename='results/summary_adni.csv'):
    data = pd.read_csv(filename)

    # get timestamp and create a folder with that name
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_summary_dir = f'results_summary/' + timestamp
    os.makedirs(results_summary_dir, exist_ok=True)

    # Sort the DataFrame by 'method', 'name', and 'seed'
    sorted_data = data.sort_values(by=['algo', 'name', 'seed'])

    # temporary removing DDPG and SAC results from results for regression testing
    # sorted_data = sorted_data[~sorted_data['algo'].isin(['DDPG', 'SAC'])]

    # Define the new column order (only the useful columns)
    column_order = ['algo', 'name', 'seed', 'epochs', 'batch_size', 'train_mae', 'test_mae', 'train_mse', 'test_mse', 'train_reward_rl', 'test_reward_rl']

    # Rearrange the columns
    sorted_data = sorted_data[column_order]

    # Write the rearranged DataFrame with specific selected columns to a new CSV file
    sorted_data.to_csv(f'{results_summary_dir}/summary_1_sorted.csv', index=False)

    # **********************************************************************************************************************
    # Now we take a mean for all 5 seeds for each algo and name(fold)
    # Group the DataFrame by 'algo' and 'name', and calculate the mean of the numerical columns
    grouped_data = sorted_data.groupby(['algo', 'name']).mean().reset_index()

    # Round off each value to 3 decimal points
    grouped_data = grouped_data.round(3)

    # replace the seed column with the number of seeds and rename column to 'num_seeds'. Remove the seed column.
    grouped_data = grouped_data.drop(columns=['seed'])
    grouped_data['num_seeds'] = sorted_data.groupby(['algo', 'name']).nunique()['seed'][0]
    # place the num_seeds as the 3rd column
    grouped_data = grouped_data[['algo', 'name', 'num_seeds', 'epochs', 'batch_size', 'train_mae', 'test_mae', 'train_mse', 'test_mse', 'train_reward_rl', 'test_reward_rl']]

    # Write the grouped DataFrame to a new CSV file
    grouped_data.to_csv(f'{results_summary_dir}/summary_2_folds.csv', index=False)

    # **********************************************************************************************************************
    # calculate mean for all rows for each algo
    # Group the DataFrame by 'algo', and calculate the mean of the numerical columns
    mean_data = grouped_data.groupby(['algo']).mean().round(3)
    std_data = grouped_data.groupby(['algo']).std().round(3)
    mean_data = mean_data.drop('num_seeds', axis=1)
    std_columns = ['train_mae', 'test_mae', 'train_mse', 'test_mse', 'train_reward_rl', 'test_reward_rl']
    # convert mean (sd) to string and concatenate, and replace the mean_data with the mean (sd) string.
    for column in std_columns:
        mean_data[column] = mean_data[column].astype(str) + " (" + std_data[column].round(3).astype(str) + ")"
    # mean_data = mean_data.round(3)
    print('\nFinal aggregate results across folds and seeds for each method:\n', mean_data)

    # Write the mean_data DataFrame to a CSV file
    mean_data.to_csv(f'{results_summary_dir}/summary_3_algo.csv')

if __name__ == '__main__':
    post_process()