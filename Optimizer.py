from PredictionModel import main, compute_weighted_competitor_price, FlightType
import pandas as pd

# Assumptions
FC = 1000000
cost_per_flight = 50000
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
                best_profit = None
                best_avg_fare = None
                # Call the function using the DataFrame and FlightType enum
                competitor_price = compute_weighted_competitor_price(df, FlightType[flight_type], year, month)
                print(f"Copetitor Price: {competitor_price}")
                avg_fares = [competitor_price * 0.8, competitor_price * 1.2]

                for avg_fare in avg_fares:
                    pax_prediction = main(year, month, avg_fare, flight_type)
                    rev_value = revenue_fn(pax_prediction, avg_fare)
                    VC_value = VC(count_of_flights(pax_prediction, plane_capacity), cost_per_flight)
                    total_cost_value = total_cost(VC_value, FC)
                    current_profit = profit(rev_value, total_cost_value)
                    if best_profit is None or current_profit > best_profit:
                        best_profit = current_profit
                        best_avg_fare = avg_fare
                print(f"Year: {year}, Month: {month}, Flight Type: {flight_type}, Best Avg Fare: {best_avg_fare}, Profit: {best_profit}")