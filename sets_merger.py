import pandas as pd

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
    'LF': 'LF_D'  # Note this rename for Load Factor
})

# Create new columns for international flights
aegean_i = aegean_i.rename(columns={
    'Pax': 'pax_I',
    'Seats': 'seats_I',
    'Count of Flights': 'flights_I',
    'Avg. Fare': 'avg_fare_I',
    'LF': 'LF_I'  # Note this rename for Load Factor
})

# Select only needed columns, including LF_D and LF_I
aegean_d = aegean_d[['Year', 'Month', 'pax_D', 'seats_D', 'flights_D', 'avg_fare_D', 'LF_D']]
aegean_i = aegean_i[['Year', 'Month', 'pax_I', 'seats_I', 'flights_I', 'avg_fare_I', 'LF_I']]

# Merge domestic and international data
aegean_merged = pd.merge(aegean_d, aegean_i, on=['Year', 'Month'])

# Merge with competitors data
final_df = pd.merge(aegean_merged, competitors_avg, on=['Year', 'Month'])

# Reorder columns as requested
final_df = final_df[[
    'Year', 'Month', 'pax_D', 'seats_D', 'flights_D', 'avg_fare_D', 'pax_I',
    'LF_D', 'LF_I', 'seats_I', 'flights_I', 'avg_fare_I', 'Selling Prices ', 'Capacities '
]]

# Round numeric columns to 2 decimal places
final_df = final_df.round(2)

# Save to CSV
final_df.to_csv('merged_set.csv', index=False)

# Display first few rows
print("\nFirst few rows of the final dataset:")
print(final_df.head())