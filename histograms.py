import pandas as pd
import matplotlib.pyplot as plt
import os

biomarkers = pd.read_excel("Data/biomarkers.xlsx")
covariates = pd.read_excel("Data/covariates.xlsx")

bio_time_0 = biomarkers[biomarkers["Biomarker"].str.contains("0weeks")].copy()

# extracting the patient ID number from the Biomarker column like "126-0weeks" -> 126
bio_time_0["PatientID"] = bio_time_0["Biomarker"].str.extract(r"(\d+)").astype(int)

merged_datasets = pd.merge(bio_time_0, covariates, on="PatientID", how="inner")

biomarkers = ["IL-8", "VEGF-A", "OPG", "TGF-beta-1", "IL-6", "CXCL9", "CXCL1", "IL-18", "CSF-1"]

# creating Pain_Group column based on VAS-at-inclusion values
merged_datasets["Pain_Group"] = merged_datasets["VAS-at-inclusion"].apply(
    lambda x: "High (≥5)" if x >= 5 else "Low (<5)"
)

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

# for easy iteration, converting axes to a flat array
axes = axes.flatten()

for i, b in enumerate(biomarkers):
    for group, data in merged_datasets.groupby("Pain_Group"):
        axes[i].hist(data[b], bins=12, alpha=0.7, label=group, edgecolor="black")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
    axes[i].set_title(b)
    axes[i].legend()

plt.tight_layout()
os.makedirs("Visualisations", exist_ok=True)
plt.savefig("Visualisations/biomarker_histograms.png")
plt.show()
