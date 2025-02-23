from PredictionModel import main, compute_weighted_competitor_price, FlightType
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assumptions
FC = 1000000
cost_per_flight = 5000
plane_capacity = 180

VC = lambda count_of_flights, cost_per_flight: count_of_flights * cost_per_flight
count_of_flights = lambda pax, capacity: pax / capacity
profit = lambda revenue, cost: revenue - cost - FC
total_cost = lambda VC, FC: VC + FC
revenue_fn = lambda pax, avg_fare: pax * avg_fare

if __name__ == "__main__":
    # Load the DataFrame once since we need it for the competitor price
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
                # Call the function using the DataFrame and FlightType enum
                competitor_price = compute_weighted_competitor_price(df, FlightType[flight_type], year, month)
                
                # Generate a range of average fares (using an appropriate step)
                avg_fares = np.arange(competitor_price * 0.7, competitor_price * 1.3, 1)
                
                # Lists to store data for plotting
                pax_vals = []
                fare_vals = []

                for avg_fare in avg_fares:
                    print(f"Avg Fare: {avg_fare}")
                    pax_prediction = main(year, month, avg_fare, flight_type)
                    print(f"Predicted Passengers: {pax_prediction}")
                    pax_vals.append(pax_prediction)
                    fare_vals.append(avg_fare)

                # Plot predicted passengers vs. average fare for this combination
                plt.figure()
                plt.plot(fare_vals, pax_vals, marker='o')
                plt.xlabel("Average Fare")
                plt.ylabel("Predicted Passengers")
                plt.title(f"Avg Fare vs. Predicted Passengers for {flight_type} - {month}/{year}")
                plt.grid(True)
                plt.show()