import pandas as pd

# Load all datasets
df1 = pd.read_csv("scan1_flows.csv")
df2 = pd.read_csv("dos1_flows.csv")
df3 = pd.read_csv("brute1_flows.csv")
df4 = pd.read_csv("normal1_flows.csv")
df5 = pd.read_csv("normal2_flows.csv")

# Combine all
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# Save final dataset
df.to_csv("final_dataset.csv", index=False)

print("✅ Final dataset created")