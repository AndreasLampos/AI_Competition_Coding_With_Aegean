import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Read the merged dataset
df = pd.read_csv('merged_set.csv')

# Calculate the average pax for each month
monthly_avg_pax = df.groupby('Month')['pax_D'].mean()

# Calculate the 90% confidence interval for the average pax for each month
confidence_intervals = {}
for month in monthly_avg_pax.index:
    month_data = df[df['Month'] == month]['pax_D']
    mean = month_data.mean()
    sem = stats.sem(month_data)
    ci = stats.t.interval(0.90, len(month_data)-1, loc=mean, scale=sem)
    confidence_intervals[month] = ci

# Plot the average pax with confidence intervals
plt.figure(figsize=(10, 6))
sns.pointplot(x=monthly_avg_pax.index, y=monthly_avg_pax.values, capsize=.2)
for month, ci in confidence_intervals.items():
    plt.errorbar(month, monthly_avg_pax[month], yerr=[[monthly_avg_pax[month] - ci[0]], [ci[1] - monthly_avg_pax[month]]], fmt='o', color='black')
plt.xlabel('Month')
plt.ylabel('Average Pax')
plt.title('Average Pax per Month with 90% Confidence Intervals')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('average_pax_per_month.png')
plt.show()

# Rank the months based on the average pax
monthly_avg_pax_sorted = monthly_avg_pax.sort_values(ascending=False)
month_rank = {month: rank+1 for rank, month in enumerate(monthly_avg_pax_sorted.index)}

# Add the ranking to the dataset
df['Month_Rank'] = df['Month'].map(month_rank)

# Save the updated dataset to a new CSV file
df.to_csv('merged_set.csv', index=False)

# Display the first few rows of the updated dataset
print(df.head())