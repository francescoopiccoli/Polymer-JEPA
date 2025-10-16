"""
Script to extract and aggregate results from multiple CSV files generated during training and testing
"""

import os
import pandas as pd
import re

# Folder where csv files are stored
# Run this script from ../
folder_path = './Results/experiments_paper/' 
split_type = 'Random'  # 'Random' or 'MonomerA' 

# Pattern to extract values from the filenames
# Files from the training procedure in main.py
if split_type == 'Random':
    filename_pattern = re.compile(
        r'metrics_test_PL_(?P<PL>.*?)_layer_norm_(?P<norm>\d+)_seeds_(?P<seeds>\d+)_finetune_percentage_(?P<percentage>[\d.]+)_pretraining_(?P<pretraining>.*?)_subgraph_type_(?P<subgraphtype>\d+)_nr_targets_(?P<nrtargets>\d+).csv'
    )
    #filename_pattern = re.compile(
    #    r'metrics_test_PL_(?P<PL>.*?)_layer_norm_(?P<norm>\d+)_seeds_(?P<seeds>\d+)_finetune_percentage_(?P<percentage>[\d.]+)_pretraining_(?P<pretraining>.*?)_subgraph_type_(?P<subgraphtype>\d+)_nr_targets_(?P<nrtargets>\d+)_context_size_(?P<context_size>[\d.]+)_target_size_(?P<target_size>[\d.]+).csv'
    #)
elif split_type == 'MonomerA':
    filename_pattern = re.compile(
        r'metrics_test_Split_type_MonomerA_PL_(?P<PL>.*?)_layer_norm_(?P<norm>\d+)_seeds_(?P<seeds>\d+)_finetune_percentage_(?P<percentage>[\d.]+)_pretraining_(?P<pretraining>.*?).csv'
    )

# Collect all the data
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        match = filename_pattern.match(filename)
        if match:
            # Extract metadata from the filename
            metadata = match.groupdict()
            
            # Load the CSV content
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            if split_type == 'Random':
                df = df[["R2", "RMSE"]]
                # Add metadata to each row
                for _, row in df.iterrows():
                    all_data.append({
                        'R2': row['R2'],
                        'RMSE': row['RMSE'],
                        **metadata
                    })
            elif split_type == 'MonomerA':
                df = df[["R2", "RMSE", "test_monomerA"]]
                # Add metadata to each row
                for _, row in df.iterrows():
                    all_data.append({
                        'R2': row['R2'],
                        'RMSE': row['RMSE'],
                        "test_monomerA": row["test_monomerA"],
                        **metadata
                    })

# Convert to DataFrame
final_df = pd.DataFrame(all_data)

# Remove rows where 'Metric' column contains things like 'R2' or 'RMSE'
#unwanted_metrics = ['R2', 'RMSE']
#final_df = final_df[~final_df['R2'].isin(unwanted_metrics)]

# Save to CSV
print(final_df)
final_df.to_csv(folder_path+'aldeghi_experiments_results_combined_metrics.csv', sep=';', decimal='.', index=False)


# Load the CSV file using semicolon as the delimiter.
df = pd.read_csv(folder_path+'aldeghi_experiments_results_combined_metrics.csv', delimiter=';')

# Standardize column names
df.columns = df.columns.str.strip()

# Convert R2 and RMSE to numeric
df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')

# Standardize the grouping columns.
# For booleans, first convert to string, strip, then map to bool.
df['PL'] = df['PL'].astype(str).str.strip().map({'True': True, 'False': False})
df['pretraining'] = df['pretraining'].astype(str).str.strip().map({'True': True, 'False': False})

# For 'norm' and 'percentage', ensure they are numeric.
df['norm'] = pd.to_numeric(df['norm'], errors='coerce')
df['percentage'] = pd.to_numeric(df['percentage'], errors='coerce')

# Optionally, you can also standardize 'seeds' though we won't group by it:
df['seeds'] = pd.to_numeric(df['seeds'], errors='coerce')

# Check dtypes to be sure everything is as expected.
print(df.dtypes)
#print(df.head())

# Now, group by the columns that define the experimental configuration.
group_keys = ['PL', 'norm', 'percentage', 'pretraining', 'subgraphtype', 'nrtargets']#, "context_size", "target_size"]

summary = df.groupby(group_keys, as_index=False).agg(
    R2_mean=('R2', 'mean'),
    R2_std=('R2', 'std'),
    RMSE_mean=('RMSE', 'mean'),
    RMSE_std=('RMSE', 'std')
)
summary = summary.assign(
    R2_mean=summary['R2_mean'].map('{:.4f}'.format),
    R2_std=summary['R2_std'].map('{:.4f}'.format),
    RMSE_mean=summary['RMSE_mean'].map('{:.4f}'.format),
    RMSE_std=summary['RMSE_std'].map('{:.4f}'.format)
)

# Save the aggregated summary to a CSV using semicolon as the delimiter.
summary.to_csv(folder_path+f'summary_statistics_{split_type}_CV.csv', sep=';', index=False)
# Summary per A_monomer type: 
if split_type == 'MonomerA':
    print("Generating summary statistics per MonomerA type...")
    summary_per_monomerA = df.groupby(group_keys + ['test_monomerA'], as_index=False).agg(
        R2_mean=('R2', 'mean'),
        R2_std=('R2', 'std'),
        RMSE_mean=('RMSE', 'mean'),
        RMSE_std=('RMSE', 'std')
    )

    # Split the df into separate CSVs for each monomer type
    mon_counter = 0
    for monomerA, group in summary_per_monomerA.groupby('test_monomerA'):
        mon_counter+=1
        output_filename = folder_path+f'summary_statistics_MonomerA_CV_{mon_counter}.csv'
        group.to_csv(output_filename, sep=';', index=False)
        print(f'Saved summary for {monomerA} to {output_filename}')