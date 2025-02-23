import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

def corrected_price_table_and_plot(avg_price):
    # Define days to departure and occupancy levels
    days = [360, 300, 240, 210, 180, 150, 120, 90, 60, 50, 40, 30, 20, 10, 8, 6, 4, 2, 1, 0]
    occupancy = np.arange(0.1, 1.01, 0.1)
    
    rows = []
    fare_matrix = np.zeros((len(days), len(occupancy)))
    
    # Calculate fare for each combination of days and occupancy
    for i, t in enumerate(days):
        # Calculate target average price for this day
        A_t = avg_price + 2 * avg_price * (1 - t / 360)
        # Starting price and slope factor
        P_start = A_t / 2
        gamma_t = A_t / 0.9
        for j, c in enumerate(occupancy):
            p = P_start + gamma_t * (c - 0.1)
            rows.append({
                "days_to_departure": t,
                "occupancy": c,
                "price": p
            })
            fare_matrix[i, j] = p

    # Create a wide-format DataFrame for the table
    df = pd.DataFrame(rows)
    df_wide = df.pivot(index="days_to_departure", columns="occupancy", values="price")
    df_wide.sort_index(ascending=False, inplace=True)
    df_wide.columns = [f"{c:.1f}" for c in df_wide.columns]
    
    # Print the table with formatting
    with pd.option_context('display.max_rows', None, 
                           'display.max_columns', None, 
                           'display.width', 1000,
                           'display.float_format', '{:.2f}'.format):
        print("Fare Table:")
        print(df_wide)
    
    # Create meshgrid for the 3D plot
    Days, Occupancy = np.meshgrid(days, occupancy, indexing='ij')
    
    # Create the 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Days, Occupancy, fare_matrix, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_xlabel('Days to Flight')
    ax.set_ylabel('Occupancy')
    ax.set_zlabel('Fare')
    ax.set_title('Connection between Days to Flight, Occupancy, and Fare')
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Fare')
    
    # Calculate and plot the curve (or points) where fare equals the desired average price
    # For each day, solve: A_t/2 + (A_t/0.9)*(c - 0.1) = avg_price  --> c = 0.1 + (avg_price - A_t/2)*(0.9/A_t)
    curve_days = []
    curve_occ = []
    curve_price = []  # This will be constant and equal to avg_price
    for t in days:
        A_t = avg_price + 2 * avg_price * (1 - t / 360)
        occ_val = 0.1 + (avg_price - (A_t / 2)) * (0.9 / A_t)
        # Only plot if occ_val is within the valid occupancy range
        if 0.1 <= occ_val <= 1.0:
            curve_days.append(t)
            curve_occ.append(occ_val)
            curve_price.append(avg_price)
    
    # Plot the intersection points with red markers
    ax.scatter(curve_days, curve_occ, curve_price, color='red', s=50, label=f"Fare = {avg_price}")
    ax.legend()
    
    # Annotate the desired average price on the plot
    ax.text2D(0.05, 0.95, f"Desired Avg FAre = {avg_price}", transform=ax.transAxes)
    
    plt.show()

if __name__ == '__main__':
    desired_avg_price = 64.16
    corrected_price_table_and_plot(desired_avg_price)
