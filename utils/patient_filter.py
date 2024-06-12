import pandas as pd
import numpy as np

def filter_patient(combined_df, algo, filter_feature, num_patients, num_seeds, years_of_data):
    # Get unique RIDs
    rids = np.unique(combined_df[algo]['RID'].values)

    rid_dict, base_cog, final_cog = {}, {}, {}

    # Data collection for each RID based on filter_feature
    for rid in rids:

        # Collect patient data based on MAE and final MMSE cognition score (which defines the range we wish to filter on). 
        if filter_feature == 'range':
            df_patient = combined_df[algo].loc[combined_df[algo]['RID'] == rid, 'cog_diff'].dropna()
            df_cog = combined_df[algo].loc[combined_df[algo]['RID'] == rid, 'MMSE_norm'].dropna()
            if len(df_cog)/num_seeds > years_of_data:                   # divide by num_seeds since we have 5x data (5 seeds/experiments per patient)
                rid_dict[rid] = np.abs(df_patient).values.mean()        # get the mean of the absolute values of the patient's cognition score
                base_cog[rid] = df_cog.iloc[0].item()                   # get the first value of the patient's cognition score
                final_cog[rid] = df_cog.iloc[len(df_cog) - 1].item()    # get the last value of the patient's cognition score

        # Collect patient data based on MAE. 
        elif filter_feature == 'mae':
            df_patient = combined_df[algo].loc[combined_df[algo]['RID'] == rid, 'cog_diff'].dropna()
            if len(df_patient)/num_seeds > years_of_data:
                rid_dict[rid] = np.abs(df_patient).values.mean()

        # Collect patient data based on cognition score (like MMSE, ADAS13)
        else:   
            df_patient = combined_df[algo].loc[combined_df[algo]['RID'] == rid, filter_feature].dropna()
            if len(df_patient)/num_seeds > years_of_data:
                baseline_year = df_patient.iloc[0].item()
                last_year = df_patient.iloc[len(df_patient) - 1].item()
                rid_dict[rid] = baseline_year - last_year
    
    # Patient selection. Sort the RIDs based on the filter_feature and collect the top num_patients
    if filter_feature == 'range':
        sorted_rids = [i[0] for i in sorted(rid_dict.items(), key=lambda x: x[1])]              # sort RIDs in ascending order of MAE score
        sorted_rids = [rid for rid in sorted_rids if 8 > final_cog[rid] > 6][:num_patients]     # filter RIDs with final cognition score between 6 and 8 (Mean MMSE=9.42, ADAS13=8.54)
        for rid in sorted_rids: print(f'Evaluating RID:{rid}, MAE:{np.round(rid_dict[rid], 3)}, base_cog:{np.round(base_cog[rid], 2)}, final_cog:{np.round(final_cog[rid], 2)}, cog_diff:{np.round(base_cog[rid] - final_cog[rid], 2)}')
    elif filter_feature == 'mae':
        sorted_rids = [i[0] for i in sorted(rid_dict.items(), key=lambda x: x[1])[:num_patients]]               # ascending order, lowest MAE first
        for rid in sorted_rids: print(f'Evaluating RID:{rid}, MAE:{np.round(rid_dict[rid], 3)}')
    else:
        sorted_rids = [i[0] for i in sorted(rid_dict.items(), key=lambda x: x[1], reverse=True)[:num_patients]] # descending order, highest cognitive difference first
        for rid in sorted_rids: print(f'Evaluating RID:{rid}, Cognition Difference:{np.round(rid_dict[rid], 2)}')
    
    return sorted_rids