import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the datasets
aegean_set = pd.read_csv('aegean_set.csv')
competitors_set = pd.read_csv('competitors_set.csv')

# Print column names to check the correct names
print("Competitors dataset columns:", competitors_set.columns.tolist())

# Filter out A3-OA and calculate averages for competitors
competitors_filtered = competitors_set[competitors_set['Carrier New'] != 'A3-OA']
competitors_avg = competitors_filtered.groupby(['Year', 'Month'])[['Selling Prices ', 'Capacities ']].mean().reset_index()

# Create separate dataframes for domestic and international flights
aegean_d = aegean_set[aegean_set['D/I'] == 'D'].copy()
aegean_i = aegean_set[aegean_set['D/I'] == 'I'].copy()

# Convert LF to numeric
aegean_d['LF'] = aegean_d['LF'].str.rstrip('%').astype(float)
aegean_i['LF'] = aegean_i['LF'].str.rstrip('%').astype(float)

# Create new columns for domestic flights
aegean_d = aegean_d.rename(columns={
    'Pax': 'pax_D',
    'Seats': 'seats_D',
    'Count of Flights': 'flights_D',
    'Avg. Fare': 'avg_fare_D',
    'LF': 'LF_D'
})

# Create new columns for international flights
aegean_i = aegean_i.rename(columns={
    'Pax': 'pax_I',
    'Seats': 'seats_I',
    'Count of Flights': 'flights_I',
    'Avg. Fare': 'avg_fare_I',
    'LF': 'LF_I'
})

# Select only needed columns, including LF_D and LF_I
aegean_d = aegean_d[['Year', 'Month', 'pax_D', 'seats_D', 'flights_D', 'avg_fare_D', 'LF_D']]
aegean_i = aegean_i[['Year', 'Month', 'pax_I', 'seats_I', 'flights_I', 'avg_fare_I', 'LF_I']]

# Merge domestic and international data
aegean_merged = pd.merge(aegean_d, aegean_i, on=['Year', 'Month'])

# Merge with competitors data
final_df = pd.merge(aegean_merged, competitors_avg, on=['Year', 'Month'])

# Calculate the average pax for each month
monthly_avg_pax = final_df.groupby('Month')['pax_D'].mean()

# Calculate the 90% confidence interval for the average pax for each month
confidence_intervals = {}
for month in monthly_avg_pax.index:
    month_data = final_df[final_df['Month'] == month]['pax_D']
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
final_df['Month_Rank'] = final_df['Month'].map(month_rank)

# Calculate the average airplane capacity for each month for domestic flights
final_df['average_airplane_capacity_D'] = final_df['seats_D'] / final_df['flights_D']

#Calculate the average airplane capacity for each month for international flights
final_df['average_airplane_capacity_I'] = final_df['seats_I'] / final_df['flights_I']

# Calculate the overall average airplane capacity for domestic flights
average_airplane_capacity_D = final_df['average_airplane_capacity_D'].mean()

#Calculate the overall average airplane capacity for international flights
average_airplane_capacity_I = final_df['average_airplane_capacity_I'].mean()

# Print the results 
print("\nAverage Airplane Capacity for Each Month for Domestic Flights:")
print(final_df[['Year', 'Month', 'average_airplane_capacity_D']])

print("\nAverage Airplane Capacity for Each Month for International Flights:")
print(final_df[['Year', 'Month', 'average_airplane_capacity_I']])

print(f"\nOverall Average Airplane Capacity (Domestic): {average_airplane_capacity_D:.2f}")
print(f"Overall Average Airplane Capacity (International): {average_airplane_capacity_I:.2f}")

# Reorder columns as requested
final_df = final_df[[
    'Year', 'Month', 'pax_D', 'seats_D', 'flights_D', 'avg_fare_D', 'pax_I',
    'LF_D', 'LF_I', 'seats_I', 'flights_I', 'avg_fare_I', 'Selling Prices ', 'Capacities ', 'Month_Rank'
]]

# Round numeric columns to 2 decimal places
final_df = final_df.round(2)

# Save to CSV
final_df.to_csv('merged_set.csv', index=False)

# Display first few rows
print("\nFirst few rows of the final dataset:")
print(final_df.head())