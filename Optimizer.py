from PredictionModel import main, compute_lag_feature, compute_lag_seats, FlightType
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assumptions
FC = 1000000
cost_per_flight = 5000
plane_capacity = 180

VC = lambda count_of_flights, cost_per_flight: count_of_flights * cost_per_flight
count_of_flights = lambda pax, capacity: pax / capacity
revenue_fn = lambda pax, avg_fare: pax * avg_fare

if __name__ == "__main__":
    # Load the dataframe once since we need it for lag features
    df = pd.read_csv("Data Files/deepthink_data_v2.csv")
    df['year'] = df['year'].astype(int)
    df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

    # Example usage
    years = [2025]
    months = [8]
    flight_types = ["DOMESTIC"]

    for year in years:
        for month in months:
            for flight_type in flight_types:
                competitor_price = compute_lag_feature(df, FlightType[flight_type], year, month)
                weighted_seats = compute_lag_seats(df, FlightType[flight_type], year, month)

                # Generate a range of average fares (using an appropriate step)
                avg_fares = np.arange(competitor_price * 0.7, competitor_price * 1.3, 1)
                
                # Initialize lists to store predicted pax, profit, and avg fares
                pax_vals = []
                profit_vals = []
                fare_vals = []

                for avg_fare in avg_fares:
                    pax_prediction = main(year, month, avg_fare, flight_type)
                    
                    # Compute profit
                    revenue = revenue_fn(pax_prediction, avg_fare)
                    flights = count_of_flights(pax_prediction, plane_capacity)
                    vc_value = VC(flights, cost_per_flight)
                    total_cost_value = vc_value + FC
                    profit_value = revenue - total_cost_value
                    
                    pax_vals.append(pax_prediction)
                    profit_vals.append(profit_value)
                    fare_vals.append(avg_fare)

                # Identify the avg fare that maximizes profit
                max_profit_index = np.argmax(profit_vals)
                sweetspot_pax = pax_vals[max_profit_index]
                sweetspot_avgfare = fare_vals[max_profit_index]
                # Set seats as 20% over the predicted pax from the sweet spot
                seats = sweetspot_pax * 1.2

                # Recalculate LF for all avg fares using this new seats value
                lf_vals = [p / seats if seats else 0 for p in pax_vals]
                # Clip LF values: any LF greater than 1 is set to 1
                lf_vals_1 = [min(lf, 1) for lf in lf_vals]

                # Get the LF at the sweet spot
                sweetspot_lf = lf_vals_1[max_profit_index]
                # Get the profit at the sweet spot
                sweetspot_profit = profit_vals[max_profit_index]

                # Print sweetspot information
                print(f"Sweetspot Avg Fare: {sweetspot_avgfare}, Pax: {sweetspot_pax}, LF: {sweetspot_lf}")

                # Plot the three subplots in one figure (stacked vertically)
                fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
                fig.suptitle(f"{flight_type} - {month}/{year}")

                # Plot LF
                axs[0].plot(fare_vals, lf_vals_1, marker='o')
                axs[0].set_ylabel("LF (Predicted Pax / Seats)")
                axs[0].grid(True)
                axs[0].axvline(sweetspot_avgfare, color='blue', linestyle='--')
                # Add legend with LF and Pax values only
                axs[0].legend([f"LF: {sweetspot_lf:.2f}"])

                # Plot Profit
                axs[1].plot(fare_vals, profit_vals, marker='o', color='green')
                axs[1].set_ylabel("Profit")
                axs[1].grid(True)
                axs[1].axvline(sweetspot_avgfare, color='blue', linestyle='--')
                # Add legend with Profit and Pax values only
                axs[1].legend([f"Profit: {sweetspot_profit:,.0f}"])

                # Plot Predicted Pax
                axs[2].plot(fare_vals, pax_vals, marker='o', color='red')
                axs[2].set_ylabel("Predicted Pax")
                axs[2].set_xlabel("Average Fare")
                axs[2].grid(True)
                axs[2].axvline(sweetspot_avgfare, color='blue', linestyle='--')
                # Add legend with Pax value only
                axs[2].legend([f"Pax: {sweetspot_pax:,.0f}"])

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()