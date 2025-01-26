import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

# Step 1: Generating sample data
# create 50 randomly chosen values from a normal distribution (mean=2.48, std dev=0.500)
diameters_sample1 = np.random.normal(2.48, 0.500, 50)
diameters_sample1_df = pd.DataFrame(diameters_sample1, columns=['diameters']).round(2)

# create 50 randomly chosen values from a normal distribution (mean=2.50, std dev=0.750)
diameters_sample2 = np.random.normal(2.50, 0.750, 50)
diameters_sample2_df = pd.DataFrame(diameters_sample2, columns=['diameters']).round(2)

# Display the first 5 observations from both samples
print("Diameters data frame of the first sample (showing only the first five observations):")
print(diameters_sample1_df.head())
print()
print("Diameters data frame of the second sample (showing only the first five observations):")
print(diameters_sample2_df.head())
print()

# Step 2: Performing hypothesis test for the difference in population proportions
# number of observations in the first sample with diameter values less than 2.20
count1 = len(diameters_sample1_df[diameters_sample1_df['diameters'] < 2.20])

# number of observations in the second sample with diameter values less than 2.20
count2 = len(diameters_sample2_df[diameters_sample2_df['diameters'] < 2.20])

# Counts Python list
counts = [count1, count2]

# Number of observations in the first and second sample
n1 = len(diameters_sample1_df)
n2 = len(diameters_sample2_df)

# n Python list
n = [n1, n2]

# Perform the hypothesis test
test_statistic, p_value = proportions_ztest(counts, n)

# Output the results
print("test-statistic =", round(test_statistic, 2))
print("two-tailed p-value =", round(p_value, 4))

# Step 3: Graphing the results
# Plot distributions of the two samples
plt.figure(figsize=(12, 6))
plt.hist(diameters_sample1, bins=15, alpha=0.7, label='Sample 1 (Mean=2.48, SD=0.5)', color='blue')
plt.hist(diameters_sample2, bins=15, alpha=0.7, label='Sample 2 (Mean=2.50, SD=0.75)', color='orange')

# Add labels and legend
plt.axvline(x=2.20, color='red', linestyle='--', label='Threshold (2.20)')
plt.title('Distribution of Diameters for Two Samples')
plt.xlabel('Diameter')
plt.ylabel('Frequency')
plt.legend()
plt.grid()

# Display the plot
plt.show()

# Visualize the test results
plt.figure(figsize=(8, 4))
x = ['Sample 1', 'Sample 2']
y = [count1 / n1, count2 / n2]  # Proportions of values below 2.20
plt.bar(x, y, color=['blue', 'orange'], alpha=0.7)
plt.title('Proportion of Diameters Below 2.20')
plt.ylabel('Proportion')
plt.ylim(0, max(y) + 0.1)
plt.grid(axis='y')

# Display the bar chart
plt.show()
