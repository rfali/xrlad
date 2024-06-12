import pandas as pd
import os

def print_stats_adni(dir_path = 'dataset/adni', num_folds=5, print_test_split_only=False):

    # Define the file names
    file_names = [os.path.join(dir_path, f'adni_fold{i}') for i in range(num_folds)]

    # Process each file
    for file_name in file_names:
        # Read all sheets from the file into a dictionary of DataFrames
        dfs = pd.read_excel(file_name + '.xls', sheet_name=None)

        # Process each sheet
        for sheet_name, df in dfs.items():
            
            # Skip printing stats for the train and valid splits if print_test_split_only is True
            if print_test_split_only and sheet_name != 'test':
                continue 
                
            # Get the unique patient Record IDs (RIDs)
            unique_RIDs = df['RID'].unique()

            # Get number of patient data entries for each year
            counts = df.groupby(['Years']).size().to_dict()

            # Get how many patients exist for each category of baseline diagnosis
            classification = df.loc[df['Years'] == 0].groupby(['DX_bl']).size().to_dict()

            # Get patient data by gender
            gender = df.loc[df['Years'] == 0].groupby(['PTGENDER']).size().to_dict()

            # Get patient data by gender
            apoe_gene = df.loc[df['Years'] == 0].groupby(['APOEPOS']).size().to_dict()

            # Get patient data by gender
            education = df.loc[df['Years'] == 0].groupby(['PTEDUCAT']).size().to_dict()

            # Print statistics about the sheet
            print(f'Fold/Split: {file_name}/{sheet_name}')
            print(f'Number of rows: {len(df)}')
            print(f'Number of unique RIDs: {len(unique_RIDs)}')
            print("RIDs : " + " ".join(map(str, unique_RIDs)))
            print(f'Number of patient entries by year: {counts}')
            print(f'Number of patient entries by baseline diagnosis: {classification}')
            print(f'Number of patient entries by gender: {gender}')
            print(f'Number of patient entries by APOE gene (0:Not_Present, 1:Present): {apoe_gene}')
            print(f'Number of patient entries by years of education: {education}')
            print('------------------------------------------------------------------------------------------------------------------------')

        print('------------------------------------------------------------------------------------------------------------------------')

if __name__ == '__main__':
    print_stats_adni()