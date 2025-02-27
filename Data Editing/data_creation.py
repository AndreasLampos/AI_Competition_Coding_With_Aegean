import pandas as pd
import numpy as np
from random import uniform, randint

# Define months and seasonal multipliers based on sample data averages
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
# Monthly passenger multipliers (normalized from sample data proportions)
pax_D_multipliers = [0.48, 0.51, 0.47, 0.64, 0.79, 1.24, 1.10, 1.43, 0.93, 0.78, 0.84, 0.80]
pax_I_multipliers = [0.29, 0.38, 0.41, 0.54, 0.92, 0.89, 1.43, 1.11, 1.30, 0.90, 0.93, 0.90]
# Load factor averages from sample data
lf_D_means = [76.5, 74.0, 66.5, 74.0, 64.5, 83.5, 81.0, 80.0, 89.5, 66.0, 87.5, 81.0]
lf_I_means = [57.0, 74.5, 78.0, 78.0, 81.5, 71.5, 89.5, 79.5, 89.5, 73.0, 88.5, 91.5]

# Function to generate data for a given year
def generate_year_data(year):
    data = []
    # Base values for 2000 (small, since operations just started)
    base_pax_D = 50000 if year >= 2000 else 0
    base_pax_I = 20000 if year >= 2000 else 0
    base_flights_D = 300 if year >= 2000 else 0
    base_flights_I = 100 if year >= 2000 else 0
    
    # Growth factor (15% CAGR, adjusted for early years)
    growth_factor = 1.15 ** (year - 2000) if year > 2000 else 1.0
    if year == 2000:
        growth_factor = 0.1  # Minimal operations in first year
    
    for i, month in enumerate(months):
        # Passengers with seasonal variation and growth
        pax_D = int(base_pax_D * growth_factor * pax_D_multipliers[i] * uniform(0.9, 1.1))
        pax_I = int(base_pax_I * growth_factor * pax_I_multipliers[i] * uniform(0.9, 1.1))
        
        # Load factors with slight variation
        lf_D = min(100, max(50, lf_D_means[i] + uniform(-5, 5)))
        lf_I = min(100, max(50, lf_I_means[i] + uniform(-5, 5)))
        
        # Seats calculated from passengers and load factor
        seats_D = int(pax_D / (lf_D / 100))
        seats_I = int(pax_I / (lf_I / 100))
        
        # Flights with growth and variation
        flights_D = int(base_flights_D * growth_factor * pax_D_multipliers[i] * uniform(0.8, 1.2))
        flights_I = int(base_flights_I * growth_factor * pax_I_multipliers[i] * uniform(0.8, 1.2))
        
        # Average fares (increase with time, higher in summer)
        fare_growth = 1.03 ** (year - 2000)  # 3% annual inflation
        avg_fare_D = round(25 + (i * 2) + uniform(-5, 5) * fare_growth, 2)
        avg_fare_I = round(60 + (i * 5) + uniform(-10, 10) * fare_growth, 2)
        
        # Competitor prices (1.5x avg fare as per sample pattern)
        comp_price_D = round(avg_fare_D * 1.5, 2)
        comp_price_I = round(avg_fare_I * 1.5, 2)
        
        # Temporary list for month ranking
        data.append({
            "year": year,
            "month": month,
            "pax_D": pax_D,
            "seats_D": seats_D,
            "flights_D": flights_D,
            "avg_fare_D": avg_fare_D,
            "pax_I": pax_I,
            "LF_D": lf_D,
            "LF_I": lf_I,
            "seats_I": seats_I,
            "flights_I": flights_I,
            "avg_fare_I": avg_fare_I,
            "competitors_price_D": comp_price_D,
            "competitors_price_I": comp_price_I,
            "month_rank": 0  # Placeholder
        })
    
    # Assign month ranks based on total passengers (pax_D + pax_I)
    df_temp = pd.DataFrame(data)
    df_temp["total_pax"] = df_temp["pax_D"] + df_temp["pax_I"]
    df_temp["month_rank"] = df_temp["total_pax"].rank(ascending=False).astype(int)
    return df_temp.drop(columns=["total_pax"]).to_dict("records")

# Generate data for 2000–2023
all_data = []
for year in range(2000, 2024):
    all_data.extend(generate_year_data(year))

# Create DataFrame and save to CSV
df = pd.DataFrame(all_data)
df = df[["year", "month", "pax_D", "seats_D", "flights_D", "avg_fare_D", "pax_I", 
         "LF_D", "LF_I", "seats_I", "flights_I", "avg_fare_I", 
         "competitors_price_D", "competitors_price_I", "month_rank"]]
df.to_csv("aegean_airlines_data_2000_2023.csv", index=False)
print("Data generated and saved to 'aegean_airlines_data_2000_2023.csv'")