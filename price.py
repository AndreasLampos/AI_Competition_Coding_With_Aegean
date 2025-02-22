import numpy as np
import pandas as pd


def logistic_price(t,c, alpha=100, delta=100, k=0.05, t0=150,gamma=80):
    """
    Logistic S-curve for airline price:
    
    P(t) = alpha + (delta / (1 + exp(-k * (t - t0))))
    
    Parameters
    ----------
    t : float or array-like
        Days to departure. If t=0 is departure day, then t>0 is days before departure.
    alpha : float
        Baseline (lower bound) of the price.
    delta : float
        Amplitude of the logistic curve above alpha. 
        So alpha + delta would be the upper asymptote of the curve.
    k : float
        Controls the steepness of the logistic growth.
    t0 : float
        Midpoint (in days). The price transitions most rapidly around t = t0.
    
    Returns
    -------
    float or ndarray
        The logistic-based price at day(s) t.
    """
    return alpha + delta / (1.0 + np.exp(k * (t - t0))) + gamma * c

def logistic_price_table():
    days = [360, 300, 240, 210, 180, 150, 120, 90, 60, 50, 40, 30, 20, 10, 8, 6, 4, 2, 1, 0]
    capacity_values = np.arange(0.1, 1.01, 0.1)
    
    rows = []
    for d in days:
        for c in capacity_values:
            p = logistic_price(d, c, 
                       alpha=100,  # baseline
                       delta=100,  # logistic amplitude
                       k=0.05,     # steepness
                       t0=150,     # midpoint in days
                       gamma=80)   # capacity sensitivity
            rows.append({
                "days_to_departure": d,
                "occupancy": c,
                "price": p
                })

    df = pd.DataFrame(rows)
    #Sort by days descending 
    df_wide = df.pivot(
        index="days_to_departure", 
        columns="occupancy", 
        values="price"
    )

    # 2) (Optional) sort days descending so that 360 appears at the top, 0 at bottom
    df_wide.sort_index(ascending=False, inplace=True)

    # 3) Print the entire wide table (disable truncation)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_wide)