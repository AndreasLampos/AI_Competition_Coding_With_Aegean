import numpy as np
import pandas as pd

def corrected_price_table(avg_price):
    days = [360, 300, 240, 210, 180, 150, 120, 90, 60, 50, 40, 30, 20, 10, 8, 6, 4, 2, 1, 0]
    capacity_values = np.arange(0.1, 1.01, 0.1)
    
    rows = []
    for t in days:
        # Target average price
        A_t = avg_price + 2 * avg_price * (1 - t / 360)
        # Starting price
        P_start = A_t / 2
        # Slope factor
        gamma_t = A_t / 0.9
        for c in capacity_values:
            p = P_start + gamma_t * (c - 0.1)
            rows.append({
                "days_to_departure": t,
                "occupancy": c,
                "price": p
            })
    
    df = pd.DataFrame(rows)
    
    # Pivot to wide format
    df_wide = df.pivot(index="days_to_departure", columns="occupancy", values="price")
    df_wide.sort_index(ascending=False, inplace=True)
    df_wide.columns = [f"{c:.1f}" for c in df_wide.columns]
    
    # Print table
    with pd.option_context('display.max_rows', None, 
                           'display.max_columns', None, 
                           'display.width', 1000,
                           'display.float_format', '{:.2f}'.format):
        print(df_wide)

if __name__ == '__main__':
    desired_avg_price = 80
    corrected_price_table(desired_avg_price)