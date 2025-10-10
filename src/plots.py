import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### DIBLOCK DATA SCENARIOS ###
plt.figure(0, figsize=(8, 5))

# Original result data from the table
ft_size = [4, 8, 12, 16, 24, 32, 48, 80]
auprc_no_prtrn = np.array([0.36, 0.44, 0.50, 0.53, 0.60, 0.65, 0.68, 0.71])
auprc_prtrn = np.array([0.40, 0.50, 0.57, 0.61, 0.65, 0.67, 0.70, 0.72])

std_dev_no_prtrn = np.array([0.03, 0.01, 0.03, 0.03, 0.02, 0.03, 0.02, 0.02])
std_dev_prtrn = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.03])

plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel("AUPRC", fontsize=20)

# # Plotting lines with standard deviation areas again

plt.plot(ft_size, auprc_prtrn, label='Jepa (wD-MPNN) - pretrained', color='green')
plt.fill_between(ft_size, auprc_prtrn - std_dev_prtrn, auprc_prtrn + std_dev_prtrn, color='green', alpha=0.05)
plt.plot(ft_size, auprc_no_prtrn, label='wD-MPNN - no pretraining', color='blue')
plt.fill_between(ft_size, auprc_no_prtrn - std_dev_no_prtrn, auprc_no_prtrn + std_dev_no_prtrn, color='blue', alpha=0.05)

plt.xticks(ft_size, [f"{size}%" for size in ft_size], rotation=45, fontsize=18)  # Converts size to percentage strings
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.savefig('Results/experiments_paper/diblock_comparison.png', dpi=300, bbox_inches='tight')


# Updated values after cross validation
""" ft_size_aldeghi_comparison = [0.4, 0.8, 1.6, 4]  # Finetune sizes in %
r2_wDMPNN_only_encoder_prtrn = np.array([0.67, 0.76, 0.86, 0.93])
r2_wDMPNN_with_mw_prtrn = np.array([0.73, 0.82, 0.87, 0.94])
r2_rf_no_prtrn = np.array([0.87, 0.87, 0.88, 0.89])
# r2_no_prtrn_ea = np.array([0.46, 0.71, 0.83, 0.94]) # , 0.96, 0.98, 0.99
r2_gao_prtrn_ = np.array([0.695654, 0.763982, 0.852779, 0.954708])
# r2_gao_baseline = np.array([0.63, 0.69,	0.80, 0.95])
r2_gao_prtrn_only_encoder = np.array([0.636246, 0.741461, 0.839005, 0.944453])

std_dev_wDMPNN_only_encoder_prtrn = np.array([0.01, 0.01, 0.02, 0.005])
std_dev_wDMPNN_with_mw_prtrn = np.array([0.03, 0.01, 0.03, 0.01])
std_dev_rf_no_prtrn = np.array([0.02, 0.02, 0.02, 0.02])
# std_dev_no_prtrn_ea = np.array([0.15, 0.06, 0.05, 0.01]) # , 0.002, 0.004, 0.002
std_dev_gao_prtrn = np.array([0.039554, 0.033191, 0.030189, 0.007161])
# std_dev_gao_baseline = np.array([0.02, 0.03, 0.02, 0.01])
std_dev_gao_prtrn_only_encoder = np.array([0.052568, 0.048467, 0.025704, 0.020673])

# Create plot for the comparison between Pretrained wD-MPNN and Random Forest not pretrained
plt.figure(2, figsize=(10, 6))
# plt.title("Pretrained wD-MPNN vs Random Forest on Aldeghi Dataset")
#plt.title("Our pretraining vs other SSL tasks on Aldeghi Dataset")
plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)

# Plotting lines with standard deviation areas
plt.plot(ft_size_aldeghi_comparison, r2_wDMPNN_only_encoder_prtrn, label='JEPA', color='green') # - Only encoder layer transfer
plt.fill_between(ft_size_aldeghi_comparison, r2_wDMPNN_only_encoder_prtrn - std_dev_wDMPNN_only_encoder_prtrn, r2_wDMPNN_only_encoder_prtrn + std_dev_wDMPNN_only_encoder_prtrn, color='green', alpha=0.05)

plt.plot(ft_size_aldeghi_comparison, r2_wDMPNN_with_mw_prtrn, label='JEPA & pseudolabel - All layers transfer', color='purple')
plt.fill_between(ft_size_aldeghi_comparison, r2_wDMPNN_with_mw_prtrn - std_dev_wDMPNN_with_mw_prtrn, r2_wDMPNN_with_mw_prtrn + std_dev_wDMPNN_with_mw_prtrn, color='purple', alpha=0.05)

# plt.plot(ft_size_aldeghi_comparison, r2_rf_no_prtrn, label='Random Forest - No pretraining', color='red')
# plt.fill_between(ft_size_aldeghi_comparison, r2_rf_no_prtrn - std_dev_rf_no_prtrn, r2_rf_no_prtrn + std_dev_rf_no_prtrn, color='red', alpha=0.05)

# plt.plot(ft_size_aldeghi_comparison, r2_no_prtrn_ea, label='wD-MPNN - No Pretraining', color='blue')
# plt.fill_between(ft_size_aldeghi_comparison, r2_no_prtrn_ea - std_dev_no_prtrn_ea, r2_no_prtrn_ea + std_dev_no_prtrn_ea, color='blue', alpha=0.05)

plt.plot(ft_size_aldeghi_comparison, r2_gao_prtrn_, label='Other SSL tasks - All layers transfer', color='orange')
plt.fill_between(ft_size_aldeghi_comparison, r2_gao_prtrn_ - std_dev_gao_prtrn, r2_gao_prtrn_ + std_dev_gao_prtrn, color='orange', alpha=0.05)

# plt.plot(ft_size_aldeghi_comparison, r2_gao_baseline, label='Gao Baseline', color='purple')
# plt.fill_between(ft_size_aldeghi_comparison, r2_gao_baseline - std_dev_gao_baseline, r2_gao_baseline + std_dev_gao_baseline, color='purple', alpha=0.05)

plt.plot(ft_size_aldeghi_comparison, r2_gao_prtrn_only_encoder, label='Other SSL tasks', color='grey') #  - Only encoder layers transfer
plt.fill_between(ft_size_aldeghi_comparison, r2_gao_prtrn_only_encoder - std_dev_gao_prtrn_only_encoder, r2_gao_prtrn_only_encoder + std_dev_gao_prtrn_only_encoder, color='grey', alpha=0.05)

# Adding custom x-axis markers
plt.xticks(ft_size_aldeghi_comparison, [f"{size}%" for size in ft_size_aldeghi_comparison])

# Add legend and show plot
plt.legend(fontsize=18)
plt.grid(True)
plt.savefig('aldeghi_comparison.png', dpi=300, bbox_inches='tight') """

# ### DIBLOCK VS RANDOM FOREST IN SMALL DATASET ###
# ft_size_diblock = [4, 24, 48, 80]  # Feature set sizes in %
# auprc_wD-MPNN_prtrn = np.array([0.40, 0.65, 0.70, 0.72])
# auprc_rf_no_prtrn = np.array([0.59, 0.70, 0.71, 0.74])
# auprc_no_prtrn = np.array([0.36, 0.60, 0.68, 0.71])

# std_dev_wD-MPNN_prtrn = np.array([0.02, 0.02, 0.01, 0.03])
# std_dev_rf_no_prtrn = np.array([0.01, 0.01, 0.01, 0.01])
# std_dev_no_prtrn = np.array([0.03, 0.02, 0.02, 0.02])

# # Create plot for the comparison between Pretrained wD-MPNN and Random Forest not pretrained on Diblock dataset
# plt.figure(figsize=(10, 6))
# plt.title("Pretrained wD-MPNN vs non-pretrained wD-MPNN on Diblock Dataset")
# plt.xlabel("Finetune dataset size (%)")
# plt.ylabel("AUPRC")

# # Plotting lines with standard deviation areas
# plt.plot(ft_size_diblock, auprc_wD-MPNN_prtrn, label='wD-MPNN - Pretrained', color='green')
# plt.fill_between(ft_size_diblock, auprc_wD-MPNN_prtrn - std_dev_wD-MPNN_prtrn, auprc_wD-MPNN_prtrn + std_dev_wD-MPNN_prtrn, color='green', alpha=0.05)

# # plt.plot(ft_size_diblock, auprc_rf_no_prtrn, label='RF - No Pretraining', color='red')
# # plt.fill_between(ft_size_diblock, auprc_rf_no_prtrn - std_dev_rf_no_prtrn, auprc_rf_no_prtrn + std_dev_rf_no_prtrn, color='red', alpha=0.05)

# plt.plot(ft_size_diblock, auprc_no_prtrn, label='wD-MPNN - No Pretraining', color='blue')
# plt.fill_between(ft_size_diblock, auprc_no_prtrn - std_dev_no_prtrn, auprc_no_prtrn + std_dev_no_prtrn, color='blue', alpha=0.1)


# # Adding custom x-axis markers
# plt.xticks(ft_size_diblock, [f"{size}%" for size in ft_size_diblock])

# # Add legend and show plot
# plt.legend()
# plt.grid(True)
# plt.show()

# ### FROZEN ENCODER WEIGHTS WHEN FINETUNING - ALDEGHI EA ### 
# ft_size_aldeghi_comparison = [0.4, 0.8, 1.6, 4, 8]  # Finetune sizes in % of full dataset # , 16
# r2_pretrained = np.array([0.016, 0.1747, 0.3623, 0.4802, 0.5621]) #  0.641
# r2_no_pretrained = np.array([-0.0001, 0.1071, 0.4756, 0.73, 0.84]) # , 0.88


# std_dev_pretrained = np.array([0.04, 0.04, 0.03, 0.03, 0.01]) # , 0.02
# std_dev_no_pretrained = np.array([0.02, 0.05, 0.08, 0.01, 0.01]) # , 0.01

# # Create plot for the comparison between Pretrained wD-MPNN and Random Forest not pretrained
# plt.figure(figsize=(10, 6))
# # plt.title("Pretrained wD-MPNN vs Random Forest on Aldeghi Dataset")
# plt.title("Pretraning vs No pretraining on Aldeghi Dataset with frozen encoder weights - EA property")
# plt.xlabel("Finetune dataset size (%)")
# plt.ylabel(r"$R^2$")

# # Plotting lines with standard deviation areas
# plt.plot(ft_size_aldeghi_comparison, r2_pretrained, label='JEPA - Pretrained', color='green')
# plt.fill_between(ft_size_aldeghi_comparison, r2_pretrained - std_dev_pretrained, r2_pretrained + std_dev_pretrained, color='green', alpha=0.05)

# plt.plot(ft_size_aldeghi_comparison, r2_no_pretrained, label='JEPA - Not pretrained', color='purple')
# plt.fill_between(ft_size_aldeghi_comparison, r2_no_pretrained - std_dev_no_pretrained, r2_no_pretrained + std_dev_no_pretrained, color='purple', alpha=0.05)

# # Adding custom x-axis markers
# plt.xticks(ft_size_aldeghi_comparison, [f"{size}%" for size in ft_size_aldeghi_comparison])

# # Add legend and show plot
# plt.legend()
# plt.grid(True)
# plt.show()

""" New plots with updated experiments from csv files. """
# Load the CSV
df = pd.read_csv('Results/experiments_paper/summary_statistics.csv', sep=';')
df_other_paper = pd.read_csv('Results/experiments_paper/summary_statistics_Gao.csv')

# Convert 'TRUE'/'FALSE' strings to booleans if needed
df['PL'] = df['PL'].astype(bool)
df['pretraining'] = df['pretraining'].astype(bool)

# Grouping data by percentage and a desired configuration (e.g. PL=True, pretraining=True)
# Different subsets
plt.figure(3,figsize=(8, 5))
df_PL_x_PT_0_N_0 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['pretraining'] == False) & (df['norm'] == 0)].sort_values(by='percentage')
df_PL_x_PT_0_N_1 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.1) & (df['pretraining'] == False) & (df['norm'] == 1)].sort_values(by='percentage')
df_PL_1_PT_1_N_0 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['PL'] == True) & (df['pretraining'] == True) & (df['norm'] == 0)].sort_values(by='percentage')
df_PL_1_PT_1_N_1 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.1) & (df['PL'] == True) & (df['pretraining'] == True) & (df['norm'] == 1)].sort_values(by='percentage')
df_PL_0_PT_1_N_0 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['PL'] == False) & (df['pretraining'] == True) & (df['norm'] == 0)].sort_values(by='percentage')
df_PL_0_PT_1_N_1 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.1) & (df['PL'] == False) & (df['pretraining'] == True) & (df['norm'] == 1)].sort_values(by='percentage')

# Match subsets with desired labels and colors
subsets = [df_PL_0_PT_1_N_0, df_PL_x_PT_0_N_0]
labels_colors = [("Jepa - pretrained","green"), ("Jepa - no pretraining (Baseline)","blue")]#, ("Jepa & pseudolabel - All layers transferred", "purple")]
#labels_colors = [("Jepa - PL","green"), ("Jepa - No PL", "green")]# ("Jepa - No pretraining", "blue"), ("Jepa - Pretrained", "orange"), ("Jepa - Only encoder layers transferred", "grey") ]
#subsets = [df_PL_x_PT_0_N_0,df_PL_x_PT_0_N_1,df_PL_1_PT_1_N_0,df_PL_1_PT_1_N_1, df_PL_0_PT_1_N_0, df_PL_0_PT_1_N_1]

for exp_sub, l_c in zip(subsets, labels_colors): 
    # Extract x and y values
    x = exp_sub['percentage']*40
    y = exp_sub['R2_mean']
    y_std = exp_sub['R2_std']
    #label = 'Polymer-JEPA (PL='+str(exp_sub["PL"].iloc[0])+', PT='+str(exp_sub["pretraining"].iloc[0])+')'
    #label = str(exp_sub["PL"].iloc[0])+str(exp_sub["pretraining"].iloc[0])+str(exp_sub["norm"].iloc[0])
    # Plot with shaded standard deviation
    if "No PL" in l_c[0]:
        plt.plot(x, y, label=l_c[0], color=l_c[1], linestyle='--')
    else: 
        plt.plot(x, y, label=l_c[0], color=l_c[1])

    plt.fill_between(x, y - y_std, y + y_std, alpha=0.1, color=l_c[1])

# Arrays from your data
new_values = np.array([0.686098, 0.794097, 0.862774, 0.930719, 0.965658])
old_values = np.array([0.490838, 0.726818, 0.846031, 0.917842, 0.961627])

# Calculate percentage improvement
percentage_improvement = ((new_values - old_values) / old_values) * 100

# Print results
for i, pct in enumerate(percentage_improvement):
    print(f"Improvement {i+1}: {pct:.2f}%")

# Results of the other paper, Only_enc_transfer
df_no_SS = df_other_paper[df_other_paper['source'] == 'No_SS'].sort_values(by='percentage')
df_no_SS['source'] = df_no_SS['source'].replace({'No_SS':'baseline'})
df_only_enc_transfer = df_other_paper[df_other_paper['source'] == 'N-SSL'].sort_values(by='percentage')
df_all_layer_transfer = df_other_paper[df_other_paper['source'] == 'NG-SSL'].sort_values(by='percentage')

subsets_other_paper = [df_all_layer_transfer, df_only_enc_transfer]#, df_only_enc_transfer]#, df_only_enc_transfer, df_no_SS]#, df_only_enc_transfer, df_all_layer_transfer]
labels_colors_other_paper = [("Gao et al. - NG-SSL (best)", "magenta"), ("Gao et al. - No PL", "magenta")]#, ("Gao et al. - Baseline", "red")]
#labels_colors_other_paper = [("Gao et al. - NG-SSL (best)", "magenta")]


for exp_sub, l_c in zip(subsets_other_paper, labels_colors_other_paper):
    x = exp_sub['percentage']*40
    y = exp_sub['mean_R2']
    y_std = exp_sub['std_R2']
    #label = "Gao et al. ("+str(exp_sub["source"].iloc[0])+')'
    # Plot with shaded standard deviation
    if "No PL" in l_c[0]:
        plt.plot(x, y, label=l_c[0], color=l_c[1], linestyle = "--")
    else: 
        plt.plot(x, y, label=l_c[0], color=l_c[1])
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.1, color=l_c[1])


plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
# Adding custom x-axis markers
plt.xticks(x, [f"{size}%" for size in x])

# Add legend and show plot
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('Results/experiments_paper/comparison_JEPA_GaoBest_PL.png', dpi=300, bbox_inches='tight')
plt.close()


# Plot all precentages of Gao et al. work 
plt.figure(4,figsize=(8, 5))
df_other_paper_all = pd.read_csv('Results/experiments_paper/summary_statistics_Gao_all_perc.csv')

df_no_SS = df_other_paper_all[(df_other_paper_all['percentage'] >= 0.01) & (df_other_paper_all['percentage'] <= 0.4) & (df_other_paper_all['source'] == 'No_SS')].sort_values(by='percentage')
df_only_enc_transfer = df_other_paper_all[(df_other_paper_all['percentage'] >= 0.01) & (df_other_paper_all['percentage'] <= 0.4) & (df_other_paper_all['source'] == 'N-SSL')].sort_values(by='percentage')
df_all_layer_transfer = df_other_paper_all[(df_other_paper_all['percentage'] >= 0.01) & (df_other_paper_all['percentage'] <= 0.4) & (df_other_paper_all['source'] == 'NG-SSL')].sort_values(by='percentage')

subsets_other_paper = [df_no_SS, df_only_enc_transfer, df_all_layer_transfer]

for exp_sub in subsets_other_paper:
    # Extract x and y values
    x = exp_sub['percentage']*40
    y = exp_sub['mean_R2']
    y_std = exp_sub['std_R2']
    label = "Gao et al. ("+str(exp_sub["source"].iloc[0])+')'
    # Plot with shaded standard deviation

    plt.plot(x, y, label=label, linestyle='--')
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.1)


plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
# Adding custom x-axis markers
plt.xticks(x, [f"{size}%" for size in x])

# Add legend and show plot
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('Results/experiments_paper/Gao_all_perc_CV.png', dpi=300, bbox_inches='tight')
plt.close()


# Plot only comparison of pretraining and no pretraining with pseudolabel false 
plt.figure(5,figsize=(8, 5))
subsets = [df_PL_0_PT_1_N_0, df_PL_x_PT_0_N_0]
labels_colors = [('JEPA (wD-MPNN) - pretrained',"green"), ("wD-MPNN - no pretraining", "blue")]
for exp_sub, l_c in zip(subsets, labels_colors): 
    # Extract x and y values
    x = exp_sub['percentage']*40
    y = exp_sub['R2_mean']
    y_std = exp_sub['R2_std']
    #label = 'Polymer-JEPA (Pretraining='+str(exp_sub["pretraining"].iloc[0])+')'
    #label = str(exp_sub["PL"].iloc[0])+str(exp_sub["pretraining"].iloc[0])+str(exp_sub["norm"].iloc[0])
    # Plot with shaded standard deviation
    
    plt.plot(x, y, label=l_c[0], color=l_c[1])
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.05, color=l_c[1])


plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
# Adding custom x-axis markers
plt.xticks(x, [f"{size}%" for size in x])

# Add legend and show plot
plt.legend(fontsize=18)
plt.grid(True)
plt.savefig('Results/experiments_paper/Pretrain_nopretrain_aldeghi_EA.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot comparison of pretraining and no pretraining with pseudolabel false and the RF model 
plt.figure(6,figsize=(8, 5))
subsets = [df_PL_1_PT_1_N_0, df_PL_x_PT_0_N_0]
labels_colors = [("Jepa (wD-MPNN) - pretrained","green"), ("wD-MPNN - No pretraining", "blue")]
for exp_sub, l_c in zip(subsets, labels_colors): 
    # Extract x and y values
    x = exp_sub['percentage']*40
    y = exp_sub['R2_mean']
    y_std = exp_sub['R2_std']
    #label = 'Polymer-JEPA (Pretraining='+str(exp_sub["pretraining"].iloc[0])+')'
    #label = str(exp_sub["PL"].iloc[0])+str(exp_sub["pretraining"].iloc[0])+str(exp_sub["norm"].iloc[0])
    # Plot with shaded standard deviation
    
    plt.plot(x, y, label=l_c[0], color=l_c[1])
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.05, color=l_c[1])

df_RF = pd.read_csv("Results/experiments_paper/summary_RF_aldeghi.csv")  # Update with actual path
# Baseline line in red
x_RF = df_RF["finetune_percentage"] * 40
y_RF = df_RF["r2_mean"]
y_std_RF = df_RF["r2_std"]
# Second baseline XGBoost line in orange
df_XGB = pd.read_csv("Results/experiments_paper/summary_XGB_aldeghi_Random_split.csv")
x_XGB = df_XGB["finetune_percentage"] * 40
y_XGB = df_XGB["r2_mean"]
y_std_XGB = df_XGB["r2_std"]

plt.plot(x, y_RF, label="Random Forest - No pretraining", color="red", linestyle="-")
plt.fill_between(x_RF, y_RF - y_std_RF, y_RF + y_std_RF, alpha=0.05, color="red")
plt.plot(x, y_XGB, label="XGBoost - No pretraining", color="orange", linestyle="-")
plt.fill_between(x_XGB, y_XGB - y_std_XGB, y_XGB + y_std_XGB, alpha=0.05, color="orange")

plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
# Adding custom x-axis markers
plt.xticks(x, [f"{size}%" for size in x])

# Add legend and show plot
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('Results/experiments_paper/Pretrain_nopretrain_RF_XGB_aldeghi_EA.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(7,figsize=(8, 5))
# Data from previous experiments: 
ft_size = [4, 8, 12, 16, 24, 32, 48, 80]
auprc_no_prtrn = np.array([0.36, 0.44, 0.50, 0.53, 0.60, 0.65, 0.68, 0.71])
auprc_prtrn = np.array([0.40, 0.50, 0.57, 0.61, 0.65, 0.67, 0.70, 0.72])

std_dev_no_prtrn = np.array([0.03, 0.01, 0.03, 0.03, 0.02, 0.03, 0.02, 0.02])
std_dev_prtrn = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.03])

# # Plotting lines with standard deviation areas again

plt.plot(ft_size, auprc_prtrn, label='JEPA (wD-MPNN) - pretrained', color='green')
plt.fill_between(ft_size, auprc_prtrn - std_dev_prtrn, auprc_prtrn + std_dev_prtrn, color='green', alpha=0.05)
plt.plot(ft_size, auprc_no_prtrn, label='wD-MPNN - No pretraining', color='blue')
plt.fill_between(ft_size, auprc_no_prtrn - std_dev_no_prtrn, auprc_no_prtrn + std_dev_no_prtrn, color='blue', alpha=0.05)

df_RF = pd.read_csv("Results/experiments_paper/summary_RF_diblock.csv") 
# Baseline line in red
x_RF = df_RF["finetune_percentage"] * 100
y_RF = df_RF["prc_mean"]
y_std_RF = df_RF["prc_std"]

plt.plot(x_RF, y_RF, label="Random Forest - No pretraining", color="red", linestyle="-")
plt.fill_between(x_RF, y_RF - y_std_RF, y_RF + y_std_RF, alpha=0.05, color="red")

# XGBoost line in orange
df_XGB = pd.read_csv("Results/experiments_paper/summary_XGB_diblock.csv")
x_XGB = df_XGB["finetune_percentage"] * 100
y_XGB = df_XGB["prc_mean"]
y_std_XGB = df_XGB["prc_std"]
plt.plot(x_XGB, y_XGB, label="XGBoost - No pretraining", color="orange", linestyle="-")
plt.fill_between(x_XGB, y_XGB - y_std_XGB, y_XGB + y_std_XGB, alpha=0.05, color="orange")

plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel("AUPRC", fontsize=20)

plt.xticks(ft_size, [f"{size}%" for size in ft_size], rotation=45, fontsize=18)  # Converts size to percentage strings
plt.yticks(fontsize=18)
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('Results/experiments_paper/diblock_comparison_RF_XGB.png', dpi=300, bbox_inches='tight')
#plt.savefig('Results/experiments_paper/diblock_comparison_RF.png', dpi=300, bbox_inches='tight')

""" Plotting for monomer-based split (9 A monomers)  """
# Plot for Monomer A split to compare JEPA with and without pretraining
plt.figure(8,figsize=(8, 5))
# Load the csv
# some settings
norm=0 # normalization layer
include_baselines = True
plot_metric = 'R2'  # or 'R2'
df_monomerA = pd.read_csv(f'Results/experiments_paper/MonomerA_CV/Norm_{norm}/summary_statistics_MonomerA_CV.csv', sep=';')
df_monomerA['PL'] = df_monomerA['PL'].astype(bool)
df_monomerA['pretraining'] = df_monomerA['pretraining'].astype(bool)
# Subsets
df_monomerA_no_prtrn = df_monomerA[(df_monomerA['percentage'] >= 0.01) & (df_monomerA['percentage'] <= 0.2) & (df_monomerA['pretraining'] == False) & (df_monomerA['norm'] == norm)].sort_values(by='percentage')
df_monomerA_prtrn = df_monomerA[(df_monomerA['percentage'] >= 0.01) & (df_monomerA['percentage'] <= 0.2) & (df_monomerA['PL'] == False) & (df_monomerA['pretraining'] == True) & (df_monomerA['norm'] == norm)].sort_values(by='percentage')
subsets = [df_monomerA_prtrn, df_monomerA_no_prtrn]
labels_colors = [("JEPA (wD-MPNN) - pretrained","green"), ("wD-MPNN - no pretraining", "blue")]
for exp_sub, l_c in zip(subsets, labels_colors): 
    # Extract x and y values
    x = exp_sub['percentage']*40
    y = exp_sub[f'{plot_metric}_mean']
    y_std = exp_sub[f'{plot_metric}_std']
    plt.plot(x, y, label=l_c[0], color=l_c[1])
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.1, color=l_c[1])
    #plt.errorbar(x, y, yerr=y_std, fmt='-o', color=l_c[1], label=l_c[0], capsize=4, elinewidth=1.5, markersize=6, alpha=0.9)


if include_baselines:
    
    df_RF = pd.read_csv("Results/experiments_paper/MonomerA_CV/summary_RF_aldeghi_MonomerA_split.csv")  # Update with actual path
    # Baseline line in red
    x_RF = df_RF["finetune_percentage"] * 40
    y_RF = df_RF[f'{plot_metric.lower()}_mean']	
    y_std_RF = df_RF[f'{plot_metric.lower()}_std']
    # Second baseline XGBoost line in orange
    df_XGB = pd.read_csv("Results/experiments_paper/MonomerA_CV/summary_XGB_aldeghi_MonomerA_split.csv")
    x_XGB = df_XGB["finetune_percentage"] * 40
    y_XGB = df_XGB[f'{plot_metric.lower()}_mean']	
    y_std_XGB = df_XGB[f'{plot_metric.lower()}_std']

    plt.plot(x, y_RF, label="Random Forest - No pretraining", color="red", linestyle="-")
    plt.fill_between(x_RF, y_RF - y_std_RF, y_RF + y_std_RF, alpha=0.05, color="red")
    plt.plot(x, y_XGB, label="XGBoost - No pretraining", color="orange", linestyle="-.")
    plt.fill_between(x_XGB, y_XGB - y_std_XGB, y_XGB + y_std_XGB, alpha=0.05, color="orange")

plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(f"${plot_metric}$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
# Adding custom x-axis markers
plt.xticks(x, [f"{size}%" for size in x])

# Add legend and show plot
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig(f'Results/experiments_paper/comparison_JEPA_baselines_MonA_split_normalization_{norm}_{plot_metric}.png', dpi=300, bbox_inches='tight')
plt.close()

# Alternatively plot boxplots from the raw data (without R2 mean and R2 std)
plt.figure(9,figsize=(12, 7))
import seaborn as sns
# some settings
norm=0 # normalization layer
include_baselines = True
plot_metric = 'R2'  #'RMSE' or 'R2'
include_SSL_baseline = True

df = pd.read_csv(f'Results/experiments_paper/MonomerA_CV/Norm_{norm}/aldeghi_experiments_results_combined_metrics.csv', sep=';')
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
# Subsets
df_monomerA_no_prtrn = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['pretraining'] == False) & (df['norm'] == norm)].sort_values(by='percentage')
df_monomerA_prtrn = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['PL'] == False) & (df['pretraining'] == True) & (df['norm'] == norm)].sort_values(by='percentage')
# Multiply the percentage by 40 to convert to actual finetune dataset size in %
df_monomerA_no_prtrn['percentage'] = df_monomerA_no_prtrn['percentage'] * 40
df_monomerA_prtrn['percentage'] = df_monomerA_prtrn['percentage'] * 40
# Combine pretrained and non-pretrained subsets
df_combined = pd.concat([df_monomerA_prtrn, df_monomerA_no_prtrn])

df_combined = df_combined.copy()
df_combined["model"] = df_combined["pretraining"].map({
    True: "JEPA (wD-MPNN) - pretrained",
    False: "wD-MPNN - no pretraining"
})

# --- 2. Load and prepare baseline models ---
if include_baselines:
    # RF baseline
    df_RF = pd.read_csv("Results/experiments_paper/MonomerA_CV/RF_results_MonomerA_split_test.csv")  # update path
    df_RF = df_RF.rename(columns={
        "finetune_percentage": "percentage",
        "r2": "R2",
        "rmse": "RMSE"
    })
    # Multiply the percentage by 40 to convert to actual finetune dataset size in %
    df_RF["percentage"] = df_RF["percentage"] * 40
    df_RF["model"] = "Random Forest"
    df_RF["pretraining"] = None  # to match columns for concat
    df_XGB = pd.read_csv("Results/experiments_paper/MonomerA_CV/XGB_results_MonomerA_split_test.csv")  # update path
    df_XGB = df_XGB.rename(columns={
        "finetune_percentage": "percentage",
        "r2": "R2",
        "rmse": "RMSE"
    })
    # Multiply the percentage by 40 to convert to actual finetune dataset size in %
    df_XGB["percentage"] = df_XGB["percentage"] * 40
    df_XGB["model"] = "XGBoost"
    df_XGB["pretraining"] = None  # to match columns for concat

    # --- 3. Combine all three datasets ---
    df_all = pd.concat([df_combined, df_RF, df_XGB], ignore_index=True)

    # --- 4. Define color palette ---
    palette = {
        "JEPA (wD-MPNN) - pretrained": "green",
        "wD-MPNN - no pretraining": "blue",
        "Random Forest": "red",
        "XGBoost": "orange"}
else:
    df_all = df_combined
    palette = {
        "JEPA (wD-MPNN) - pretrained": "green",
        "wD-MPNN - no pretraining": "blue"}
    
if include_SSL_baseline: 
    df = pd.read_csv("Results/experiments_paper/MonomerA_CV/Gao_all_results_MonA_split_combined.csv", sep=';') 
    df.columns = df.columns.str.strip()
    df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
    df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')
    # Standardize the grouping columns.
    # For booleans, first convert to string, strip, then map to bool.
    # For 'norm' and 'percentage', ensure they are numeric.
    df['percentage'] = pd.to_numeric(df['percentage'], errors='coerce')
    # Optionally, you can also standardize 'seeds' though we won't group by it:
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.rename(columns={
        "time": "seeds",
    })
    df_subset_pretrain = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['label']==0) & (df['pretraining']==True)].sort_values(by='percentage')
    df_subset_no_pretrain = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['label']==0) & (df['pretraining']==False)].sort_values(by='percentage')
    # Multiply the percentage by 40 to convert to actual finetune dataset size in %
    df_subset_pretrain["percentage"] = df_subset_pretrain["percentage"] * 40
    df_subset_no_pretrain["percentage"] = df_subset_no_pretrain["percentage"] * 40
    df_subset_pretrain["model"] = "Gao et al. - NG-SSL (best)"
    df_subset_no_pretrain["model"] = "Gao et al. - No pretrain"
    #df_all = pd.concat([df_all, df_subset_pretrain, df_subset_no_pretrain], ignore_index=True)
    df_all = pd.concat([df_all, df_subset_pretrain], ignore_index=True)
    palette["Gao et al. - NG-SSL (best)"] = "purple"
    palette["Gao et al. - No pretrain"] = "pink"

print(df_all)
print(df)

sns.boxplot(
    data=df_all,
    x="percentage",
    y=plot_metric,
    hue="model",
    palette=palette,
    dodge=True,
    width=0.6,
    linewidth=1.2
)

# --- 6. Style the plot ---
plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(f"${plot_metric}$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title="Model", fontsize=14, title_fontsize=16)
plt.grid(True, axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(f"Results/experiments_paper/boxplot_JEPA_vs_Baseline_SSLBaseline_{include_SSL_baseline}_MonA_split_normalization_{norm}_{plot_metric}.png",
            dpi=300, bbox_inches="tight")

plt.close()

# Plot per monomer curves of pretraining vs. non pretraining, 9 subplots (3x3)
monomers = [
    "[*:1]c1cc(F)c([*:2])cc1F",
    "[*:1]c1cc2cc3sc([*:2])cc3cc2s1",
    "[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34",
    "[*:1]c1ccc(-c2ccc([*:2])s2)s1",
    "[*:1]c1ccc([*:2])c2nsnc12",
    "[*:1]c1ccc([*:2])cc1",
    "[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2",
    "[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2",
    "[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12"
]


# Check dissimilarity of test monomer against the rest
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
# Clean monomer smiles should not have the [*:#] attachment points when computing similarity

def replace_dummies_with_H(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Find all dummy atoms (*)
    rwmol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            idx = atom.GetIdx()
            rwmol.ReplaceAtom(idx, Chem.Atom('H'))
    
    mol_H = rwmol.GetMol()
    Chem.SanitizeMol(mol_H)
    return Chem.MolToSmiles(mol_H, canonical=True)

clean_monomers = [m.replace("[*:1]", "[*]").replace("[*:2]", "[*]") for m in monomers]
#clean_monomers = [replace_dummies_with_H(m) for m in monomers]
mols = [Chem.MolFromSmiles(m) for m in clean_monomers]
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]

# Compute average dissimilarity
sim_matrix = np.array([[DataStructs.TanimotoSimilarity(f1, f2) for f2 in fps] for f1 in fps])
avg_sim = sim_matrix.mean(axis=1)
dissimilarity = 1 - avg_sim
# Store in DataFrame
dissim_df = pd.DataFrame({
    "monomer": monomers,
    "dissimilarity": dissimilarity,
    "average_similarity": avg_sim
})
# Compute nearest-neighbor dissimilarity
results = []
for i, fp_i in enumerate(fps):
    # Compute similarity to all *other* monomers
    sims = [DataStructs.TanimotoSimilarity(fp_i, fp_j) for j, fp_j in enumerate(fps) if i != j]
    max_sim = max(sims)                     # most similar other monomer
    min_dist = 1 - max_sim                  # dissimilarity to closest monomer
    results.append({
        "monomer": clean_monomers[i],
        "most_similar_to": clean_monomers[np.argmax(sims) if sims else i],
        "similarity": max_sim,
        "dissimilarity": min_dist
    })

dissim_df_2 = pd.DataFrame(results)
print(dissim_df_2)

# Create 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

norm=0 # normalization layer
boxplot = True
plot_metric = 'R2'  #'RMSE' or 'R2'
add_SSL_baseline = True
# Load the csv with all data so we can do boxplots as well
df_all = pd.read_csv(f'Results/experiments_paper/MonomerA_CV/Norm_{norm}/aldeghi_experiments_results_combined_metrics.csv', sep=';')
for i, monomer in enumerate(monomers):
    ax = axes[i]
    if boxplot:
        # subset for the specific monomer
        df_monomer = df_all[df_all['test_monomerA'] == monomer]
    else: 
        df_monomer = pd.read_csv(f'Results/experiments_paper/MonomerA_CV/Norm_{norm}/summary_statistics_MonomerA_CV_{i+1}.csv', sep=';')
    df_monomer['PL'] = df_monomer['PL'].astype(bool)
    df_monomer['pretraining'] = df_monomer['pretraining'].astype(bool)

    # Filter subsets
    df_no_prtrn = df_monomer[
        (df_monomer['percentage'] >= 0.01) &
        (df_monomer['percentage'] <= 0.2) &
        (df_monomer['pretraining'] == False) &
        (df_monomer['norm'] == norm)
    ].sort_values(by='percentage')

    df_prtrn = df_monomer[
        (df_monomer['percentage'] >= 0.01) &
        (df_monomer['percentage'] <= 0.2) &
        (df_monomer['PL'] == False) &
        (df_monomer['pretraining'] == True) &
        (df_monomer['norm'] == norm)
    ].sort_values(by='percentage')
    # multiply the percentage by 40 to convert to actual finetune dataset size in %
    df_no_prtrn['percentage'] = df_no_prtrn['percentage'] * 40
    df_prtrn['percentage'] = df_prtrn['percentage'] * 40

    df_combined = pd.concat([df_prtrn, df_no_prtrn])
    if add_SSL_baseline:
        df = pd.read_csv("Results/experiments_paper/MonomerA_CV/Gao_all_results_MonA_split_with_monomers.csv", sep=';') 
        # Extract the subset for the specific monomer
        df = df[df['test_monomerA'] == monomer]
        df.columns = df.columns.str.strip()
        df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
        df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')
        # Standardize the grouping columns.
        # For booleans, first convert to string, strip, then map to bool.
        # For 'norm' and 'percentage', ensure they are numeric.
        df['percentage'] = pd.to_numeric(df['percentage'], errors='coerce')
        # Optionally, you can also standardize 'seeds' though we won't group by it:
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df = df.rename(columns={
            "time": "seeds",
        })
        df_subset_pretrain = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['label']==0)].sort_values(by='percentage')
        # Multiply the percentage by 40 to convert to actual finetune dataset size in %
        df_subset_pretrain["percentage"] = df_subset_pretrain["percentage"] * 40
        df_subset_pretrain["model"] = "Gao et al. - NG-SSL (best)"
        #df_all = pd.concat([df_all, df_subset_pretrain, df_subset_no_pretrain], ignore_index=True)
        df_combined = pd.concat([df_combined, df_subset_pretrain], ignore_index=True)
        palette["Gao et al. - NG-SSL (best)"] = "purple"
    if boxplot: 
        df_combined = df_combined.copy()
        df_combined["model"] = df_combined["pretraining"].map({
            True: "JEPA (wD-MPNN) - pretrained",
            False: "wD-MPNN - no pretraining"
        }).fillna("Gao et al. - NG-SSL (best)")
        sns.boxplot(
            data=df_combined,
            x="percentage",
            y=plot_metric,
            hue="model",
            palette={
                "JEPA (wD-MPNN) - pretrained": "green",
                "wD-MPNN - no pretraining": "blue",
                "Gao et al. - NG-SSL (best)": "purple"
            },
            dodge=True,
            width=0.6,
            linewidth=1.2,
            ax=ax
        )
    else: 
        if add_SSL_baseline:
            # aggregate SSL baseline for plotting
            df_SSL = df_subset_pretrain.groupby('percentage').agg(
                RMSE_mean=('RMSE', 'mean'),
                RMSE_std=('RMSE', 'std')
            ).reset_index()
            subsets = [df_prtrn, df_no_prtrn, df_SSL]
            labels_colors = [("JEPA (wD-MPNN) - pretrained", "green"),
                            ("wD-MPNN - no pretraining", "blue"),
                            ("Gao et al. - NG-SSL (best)", "purple")]
        else:
            subsets = [df_prtrn, df_no_prtrn]
            labels_colors = [("JEPA (wD-MPNN) - pretrained", "green"),
                            ("wD-MPNN - no pretraining", "blue")]
        for exp_sub, (label, color) in zip(subsets, labels_colors):
            x = exp_sub['percentage']
            y = exp_sub['RMSE_mean']
            y_std = exp_sub['RMSE_std']
            
            ax.errorbar(x, y, yerr=y_std,
                        fmt='-o', color=color, label=label,
                    capsize=4, elinewidth=1.5, markersize=6, alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:.1f}%" for v in x])

    ax.set_title(f'Monomer: {monomer}', fontsize=10)
    ax.set_xlabel("Finetune dataset size (%)", fontsize=10)
    ax.set_ylabel(r"$RMSE$", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=8)
    # add dissimilarity in the top right corner under legend
    sim_value = dissim_df[dissim_df['monomer'] == monomer]['average_similarity'].values[0]
    #sim_value = dissim_df_2[dissim_df_2['monomer'] == clean_monomers[i]]['similarity'].values[0]
    ax.text(0.95, 0.95, f'Similarity to training set: {sim_value:.2f}', transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right')
    ax.grid(True)

# Adjust layout for readability
plt.tight_layout()
plt.savefig(f'Results/experiments_paper/comparison_JEPA_baselines_SSLBaseline_{add_SSL_baseline}_per_monomer_normalization_{norm}_boxplot_{boxplot}_{plot_metric}.png', dpi=300, bbox_inches='tight')

# Do the same plot with per monomer performance for the SSL baseline
# Create 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
# full df read
df_full = pd.read_csv(f'Results/experiments_paper/MonomerA_CV/Gao_all_results_MonA_split_with_monomers.csv', sep=';')
for i, monomer in enumerate(monomers):
    ax = axes[i]
    # monomer subset from full df based on monomer name in column test_monomerA
    df_monomer = df_full[df_full['test_monomerA'] == monomer]
    df_monomer.columns = df_monomer.columns.str.strip()
    df_monomer['R2'] = pd.to_numeric(df_monomer['R2'], errors='coerce')
    df_monomer['RMSE'] = pd.to_numeric(df_monomer['RMSE'], errors='coerce')
    # Standardize the grouping columns.
    # For booleans, first convert to string, strip, then map to bool.
    # For 'norm' and 'percentage', ensure they are numeric.
    df_monomer['percentage'] = pd.to_numeric(df_monomer['percentage'], errors='coerce')
    # Optionally, you can also standardize 'seeds' though we won't group by it:
    df_monomer['time'] = pd.to_numeric(df_monomer['time'], errors='coerce')
    df_subset = df_monomer[(df_monomer['percentage'] >= 0.01) & (df_monomer['percentage'] <= 0.2) & (df_monomer['label']==0)].sort_values(by='percentage')
    df_subset = df_subset.rename(columns={
        "time": "seeds",
    })
    # Multiply the percentage by 40 to convert to actual finetune dataset size in %
    df_subset["percentage"] = df_subset["percentage"] * 40
    df_subset["model"] = "Gao et al. - NG-SSL (best)"

    # Before plotting, group and average for RMSE and R2
    df_subset = df_subset.groupby(['percentage', 'model'], as_index=False).agg(
        RMSE=('RMSE', 'mean'),
        RMSE_std=('RMSE', 'std'),
        R2=('R2', 'mean'),
        R2_std=('R2', 'std')
    )

    x = df_subset['percentage']
    y = df_subset['RMSE']
    y_std = df_subset['RMSE_std']

    #ax.plot(x, y, '-o', color='pink', label="Gao et al. - NG-SSL (best)", alpha=0.9)
    ax.errorbar(x, y, yerr=y_std)

    ax.set_title(f'Monomer: {monomer}', fontsize=10)
    ax.set_xlabel("Finetune dataset size (%)", fontsize=10)
    ax.set_ylabel(r"$RMSE$", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=8)
    # add dissimilarity in the top right corner under legend
    sim_value = dissim_df[dissim_df['monomer'] == monomer]['average_similarity'].values[0]
    #sim_value = dissim_df_2[dissim_df_2['monomer'] == clean_monomers[i]]['similarity'].values[0]
    ax.text(0.95, 0.95, f'Similarity to training set: {sim_value:.2f}', transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right')
    ax.grid(True)
# Adjust layout for readability
plt.tight_layout()
plt.savefig(f'Results/experiments_paper/Gao_baseline_per_monomer_performance.png', dpi=300, bbox_inches='tight')

### ---- Boxplot for random split ---- ###
plt.figure(11,figsize=(12, 7))
import seaborn as sns
# some settings
norm=0 # normalization layer
include_baselines = True
plot_metric = 'R2'  #'RMSE' or 'R2'
df = pd.read_csv(f'Results/experiments_paper/Random_5F_CV/aldeghi_experiments_results_combined_metrics.csv', sep=';')
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
# Subsets
df_no_prtrn = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['pretraining'] == False) & (df['norm'] == norm)].sort_values(by='percentage')
df_prtrn = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['PL'] == False) & (df['pretraining'] == True)].sort_values(by='percentage')
# Multiply the percentage by 40 to convert to actual finetune dataset size in %
df_no_prtrn['percentage'] = df_no_prtrn['percentage'] * 40
df_prtrn['percentage'] = df_prtrn['percentage'] * 40
# Combine pretrained and non-pretrained subsets
df_combined = pd.concat([df_prtrn, df_no_prtrn])
df_combined = df_combined.copy()
df_combined["model"] = df_combined["pretraining"].map({
    True: "JEPA (wD-MPNN) - pretrained",
    False: "wD-MPNN - no pretraining"
})
# --- 2. Load and prepare baseline models ---
if include_baselines:
    # RF baseline
    df_RF = pd.read_csv("Results/experiments_paper/Random_5F_CV/RF_results_test.csv")  # update path
    df_RF = df_RF.rename(columns={
        "finetune_percentage": "percentage",
        "r2": "R2",
        "rmse": "RMSE"
    })
    # Multiply the percentage by 40 to convert to actual finetune dataset size in %
    df_RF["percentage"] = df_RF["percentage"] * 40
    df_RF["model"] = "Random Forest"
    df_RF["pretraining"] = None  # to match columns for concat
    df_XGB = pd.read_csv("Results/experiments_paper/Random_5F_CV/XGB_results_Random_split_test.csv")  # update path
    df_XGB = df_XGB.rename(columns={
        "finetune_percentage": "percentage",
        "r2": "R2",
        "rmse": "RMSE"
    })
    # Multiply the percentage by 40 to convert to actual finetune dataset size in %
    df_XGB["percentage"] = df_XGB["percentage"] * 40
    df_XGB["model"] = "XGBoost"
    df_XGB["pretraining"] = None  # to match columns for concat

    # --- 3. Combine all three datasets ---
    df_all = pd.concat([df_combined, df_RF, df_XGB], ignore_index=True)

    # --- 4. Define color palette ---
    palette = {
        "JEPA (wD-MPNN) - pretrained": "green",
        "wD-MPNN - no pretraining": "blue",
        "Random Forest": "red",
        "XGBoost": "orange"}
    
else:
    df_all = df_combined
    palette = {
        "JEPA (wD-MPNN) - pretrained": "green",
        "wD-MPNN - no pretraining": "blue"}
    
if include_SSL_baseline:
    df = pd.read_csv("Results/experiments_paper/Random_5F_CV/Gao_all_results_random_split.csv", sep=';') 
    df.columns = df.columns.str.strip()
    df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
    df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')
    # Standardize the grouping columns.
    # For booleans, first convert to string, strip, then map to bool.
    # For 'norm' and 'percentage', ensure they are numeric.
    df['percentage'] = pd.to_numeric(df['percentage'], errors='coerce')
    # Optionally, you can also standardize 'seeds' though we won't group by it:
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.rename(columns={
        "time": "seeds",
    })
    # If percentage in [0.01, 0.02, 0.04, 0.1, 0.2], then add it to the plot
    filtered_df = df[df['percentage'].isin([0.01, 0.02, 0.04, 0.1, 0.2])]
    df_subset_pretrain = filtered_df[(filtered_df['percentage'] >= 0.01) & (filtered_df['percentage'] <= 0.2) & (filtered_df['label']==0)].sort_values(by='percentage')
    # Multiply the percentage by 40 to convert to actual finetune dataset size in %
    df_subset_pretrain["percentage"] = df_subset_pretrain["percentage"] * 40
    df_subset_pretrain["model"] = "Gao et al. - NG-SSL (best)"
    df_all = pd.concat([df_all, df_subset_pretrain], ignore_index=True)
    palette["Gao et al. - NG-SSL (best)"] = "purple"

sns.boxplot(
    data=df_all,
    x="percentage",
    y=plot_metric,
    hue="model",
    palette=palette,
    dodge=True,
    width=0.6,
    linewidth=1.2
)
# --- 6. Style the plot ---
plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(f"${plot_metric}$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title="Model", fontsize=14, title_fontsize=16)
plt.grid(True, axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(f"Results/experiments_paper/boxplot_JEPA_vs_Baseline_SSLBaseline_{include_SSL_baseline}_Random_split_normalization_{norm}_{plot_metric}.png",
            dpi=300, bbox_inches="tight")