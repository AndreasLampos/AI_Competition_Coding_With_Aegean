from PredictionModel import main, compute_lag_feature, compute_lag_seats, FlightType
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
                # Retrieve lagged competitor price and weighted seats using functions from PredictionModel.py
                competitor_price = compute_lag_feature(df, FlightType[flight_type], year, month)
                seats = compute_lag_seats(df, FlightType[flight_type], year, month)

                # Generate a range of average fares (using an appropriate step)
                avg_fares = np.arange(competitor_price * 0.7, competitor_price * 1.3, 1)
                
                # Initialize lists for LF (predicted pax/weighted seats), profit, and predicted pax
                lf_vals = []
                profit_vals = []
                pax_vals = []
                fare_vals = []

                i = 0
                for avg_fare in avg_fares:
                    i += 1
                    print(f"Avg Fare: {avg_fare}")
                    pax_prediction = main(year, month, avg_fare, flight_type)
                    if i == 1:
                        seats = pax_prediction
                        print(f"Seats: {seats}")
                    
                    # Compute LF = predicted pax divided by weighted seats
                    LF = pax_prediction / seats if seats else 0

                    # Compute profit:
                    # Revenue: predicted pax * avg fare.
                    revenue = revenue_fn(pax_prediction, avg_fare)
                    # Number of flights required:
                    flights = count_of_flights(pax_prediction, plane_capacity)
                    # Variable cost:
                    vc_value = VC(flights, cost_per_flight)
                    # Total cost:
                    total_cost_value = vc_value + FC
                    profit_value = revenue - total_cost_value
                    
                    print(f"Predicted Pax: {pax_prediction}, LF: {LF}, Profit: {profit_value}")
                    
                    lf_vals.append(LF)
                    profit_vals.append(profit_value)
                    pax_vals.append(pax_prediction)
                    fare_vals.append(avg_fare)

                # Plot the three subplots in one figure (stacked vertically)
                fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
                fig.suptitle(f"{flight_type} - {month}/{year}")

                axs[0].plot(fare_vals, lf_vals, marker='o')
                axs[0].set_ylabel("LF (Predicted Pax / Weighted Seats)")
                axs[0].grid(True)

                axs[1].plot(fare_vals, profit_vals, marker='o', color='green')
                axs[1].set_ylabel("Profit")
                axs[1].grid(True)

                axs[2].plot(fare_vals, pax_vals, marker='o', color='red')
                axs[2].set_ylabel("Predicted Pax")
                axs[2].set_xlabel("Average Fare")
                axs[2].grid(True)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()