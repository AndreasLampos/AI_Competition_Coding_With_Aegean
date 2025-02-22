from PredictionModel import main

if __name__ == "__main__":
    # Example usage
    year = 2025
    month = 8
    avg_fare = 300.0
    #selling_prices = 350.0
    #capacities = 150.0
    flight_type_str = 'DOMESTIC'
    prediction = main(year, month, avg_fare, selling_prices, capacities, flight_type_str)
    print(f"Predicted Number of {flight_type_str} Passengers: {prediction:.0f}")