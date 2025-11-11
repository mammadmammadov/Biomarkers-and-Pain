import pandas as pd
from scipy.stats import ttest_ind

biomarkers = pd.read_excel("Data/biomarkers.xlsx")
covariates = pd.read_excel("Data/covariates.xlsx")

bio_time_0 = biomarkers[biomarkers["Biomarker"].str.contains("0weeks")].copy()

# extracting the patient ID number from the Biomarker column like "126-0weeks" -> 126
bio_time_0["PatientID"] = bio_time_0["Biomarker"].str.extract(r"(\d+)").astype(int)

merged_datasets = pd.merge(bio_time_0, covariates, on="PatientID", how="inner")

merged_datasets["Pain_Group"] = merged_datasets["VAS-at-inclusion"].apply(
    lambda x: "High (≥5)" if x >= 5 else "Low (<5)"
)

biomarkers = [
    "IL-8", "VEGF-A", "OPG", "TGF-beta-1", "IL-6", "CXCL9", "CXCL1", "IL-18", "CSF-1"
]

# we will perform two-sample t-tests for each biomarker

results = []

# significance level
alpha = 0.05
 
for b in biomarkers:
    high = merged_datasets.loc[merged_datasets["Pain_Group"] == "High (≥5)", b].dropna()
    low = merged_datasets.loc[merged_datasets["Pain_Group"] == "Low (<5)", b].dropna()

    # getting t-statistic and p-values
    # p-value implicitly uses n1 + n2 - 2 degrees of freedom, where n1 and n2 are sample sizes of the two groups
    t_stat, p_val = ttest_ind(high, low, equal_var=False)

    results.append({
        "Biomarker": b,
        "Sample Mean for High-VAS": high.mean(),
        "Sample Mean for Low-VAS": low.mean(),
        "t-Statistic": t_stat,
        "p-Value": p_val
    })

t_results = pd.DataFrame(results)

print("\nTwo-sample t-test Results (High-VAS vs Low-VAS Groups at Inclusion)\n")
print(t_results.to_string(index=False))
