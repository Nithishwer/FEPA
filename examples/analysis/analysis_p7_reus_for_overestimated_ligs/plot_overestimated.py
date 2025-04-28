import os
import pandas as pd

output_file = os.path.join("sorted_exp_abfe_df.csv")
df = pd.read_csv(output_file)
# Creating a new column appending Ligname column to Van column
df["ABFE_name"] = [str(i) + "_" + str(j) for i, j in zip(df["Lig_Name"], df["Vanilla"])]

# Plot overestimation
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x="ABFE_name", y="overestimation", data=df, palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("ABFE Name")
plt.ylabel("Overestimation")
plt.title("Overestimation by ABFE Name")
plt.tight_layout()
plt.savefig("overestimation_by_abfe_name.png")
plt.show()
plt.close()
