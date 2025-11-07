import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

def load_matched_data():
    """
    Load the matched dataset generated from your propensity score matching.
    It is assumed to be saved as 'matched_data.csv' in the processed folder.
    """
    matched_fp = os.path.join("..", "data", "processed", "matched_data.csv")
    df = pd.read_csv(matched_fp)
    return df

def compute_att(df, outcome_col="lossyear_mean"):
    """
    Compute the ATT by grouping matched pairs.
    We assume that each treated unit is paired (or grouped) with one or more controls,
    identified by a 'matched_id' column.
    """
    grouped = df.groupby("matched_id")
    differences = []
    for group_id, group in grouped:
        # Compute average outcome for treated and controls in the group
        treated_outcome = group[group["is_protected"] == 1][outcome_col].mean()
        control_outcome = group[group["is_protected"] == 0][outcome_col].mean()
        differences.append(treated_outcome - control_outcome)
    att = np.mean(differences)
    return att, differences

def paired_ttest(df, outcome_col="lossyear_mean"):
    """
    Perform a paired t-test on the differences in outcome for each matched pair.
    This function assumes that each matched group (identified by 'matched_id') contains
    exactly one treated and one control unit.
    """
    groups = df.groupby("matched_id")
    diff_list = []
    for group_id, group in groups:
        if group["is_protected"].nunique() < 2:
            continue  # skip groups that don't have both treated and control
        treated_val = group[group["is_protected"] == 1][outcome_col].values[0]
        control_val = group[group["is_protected"] == 0][outcome_col].values[0]
        diff_list.append(treated_val - control_val)
    t_stat, p_value = stats.ttest_1samp(diff_list, 0)
    return t_stat, p_value, diff_list

def bootstrap_att(diff_list, n_boot=1000, alpha=0.05):
    """
    Compute bootstrap confidence intervals for the ATT.
    """
    diff_array = np.array(diff_list)
    boot_atts = []
    rng = np.random.default_rng(42)
    for _ in tqdm(range(n_boot), desc="Bootstrapping ATT"):
        sample = rng.choice(diff_array, size=len(diff_array), replace=True)
        boot_atts.append(np.mean(sample))
    lower = np.percentile(boot_atts, 100 * alpha / 2)
    upper = np.percentile(boot_atts, 100 * (1 - alpha / 2))
    return lower, upper

def plot_outcome_distributions(df, outcome_col="lossyear_mean"):
    """
    Create a boxplot to compare the distribution of the outcome between treated and control groups.
    """
    plt.figure(figsize=(10,6))
    sns.boxplot(x="is_protected", y=outcome_col, data=df)
    plt.title("Outcome Distribution by Protection Status")
    plt.xlabel("Protection Status (0=Unprotected, 1=Protected)")
    plt.ylabel(outcome_col)
    plt.show()

def main():
    # Load matched data
    df = load_matched_data()
    print("Matched data summary (first few rows):")
    print(df.head())
    
    # Compute ATT using the grouped differences method
    att, diff_list = compute_att(df, outcome_col="lossyear_mean")
    print(f"Estimated ATT (treated - control): {att:.4f}")
    
    # Perform a paired t-test on the differences
    t_stat, p_value, diff_list = paired_ttest(df, outcome_col="lossyear_mean")
    print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    
    # Bootstrap the ATT to obtain a confidence interval
    lower, upper = bootstrap_att(diff_list, n_boot=1000, alpha=0.05)
    print(f"Bootstrap 95% CI for ATT: [{lower:.4f}, {upper:.4f}]")
    
    # Plot outcome distributions
    plot_outcome_distributions(df, outcome_col="lossyear_mean")
    
if __name__ == "__main__":
    main()
