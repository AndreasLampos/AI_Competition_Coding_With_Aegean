import numpy as np
import pandas as pd


def logistic_price(t, c, alpha, delta, k, t0, gamma):
    """
    Logistic S-curve for airline price:
    
    P(t) = alpha + (delta / (1 + exp(-k * (t - t0)))) + gamma * c
    
    Parameters
    ----------
    t : float or array-like
        Days to departure. If t=0 is departure day, then t>0 is days before departure.
    alpha : float
        Baseline (lower bound) of the price.
    delta : float
        Amplitude of the logistic curve above alpha.
    k : float
        Controls the steepness of the logistic growth.
    t0 : float
        Midpoint (in days). The price transitions most rapidly around t = t0.
    gamma : float
        Sensitivity to occupancy.
    
    Returns
    -------
    float or ndarray
        The logistic-based price at day(s) t.
    """
    return alpha + delta / (1.0 + np.exp(k * (t - t0))) + gamma * c


def logistic_price_table(avg_price):
    """
    Computes a table of prices based on days to departure and occupancy.
    The computed prices are then shifted so that the overall average price
    equals the desired avg_price.
    
    Parameters
    ----------
    avg_price : float
        The desired overall average price for the table.
    """
    days = [360, 300, 240, 210, 180, 150, 120, 90, 60, 50, 40, 30, 20, 10, 8, 6, 4, 2, 1, 0]
    capacity_values = np.arange(0.1, 1.01, 0.1)
    
    rows = []
    for d in days:
        for c in capacity_values:
            p = logistic_price(d, c, alpha=100, delta=100, k=0.005, t0=260, gamma=80)
            rows.append({
                "days_to_departure": d,
                "occupancy": c,
                "price": p
            })
    
    df = pd.DataFrame(rows)
    
    # Compute the current overall average price of the table.
    current_avg = df['price'].mean()
    # Determine the offset needed to shift the average to the desired avg_price.
    offset = avg_price - current_avg
    # Apply the offset to all prices.
    df['price'] = df['price'] + offset
    
    # Pivot to create a wide table with days as rows and occupancy values as columns.
    df_wide = df.pivot(index="days_to_departure", columns="occupancy", values="price")
    df_wide.sort_index(ascending=False, inplace=True)
    df_wide.columns.name = None  # Remove the extra header on columns.
    
    # Print the entire table without truncation.
    with pd.option_context('display.max_rows', None, 
                           'display.max_columns', None, 
                           'display.width', 1000):
        print(df_wide)


if __name__ == '__main__':
    # Set the desired average price.
    desired_avg_price = 80  # Change this value as needed.
    logistic_price_table(desired_avg_price)
