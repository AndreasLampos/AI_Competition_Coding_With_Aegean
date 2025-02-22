from PredictionModel import main


#Assumptions
FC = 1000000
cost_per_flight = 50000
plane_capacity = 180

VC = lambda count_of_flights,cost_per_flight: count_of_flights * cost_per_flight

count_of_flights = lambda pax, capacity: pax / capacity

profit = lambda revenue, cost: revenue - cost - FC

total_cost = lambda VC, FC: VC + FC

revenue = lambda pax, avg_fare: pax * avg_fare


if __name__ == "__main__":
    # Example usage
    years = [2025]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    avg_fares = [10,250]
    flight_types = ["DOMESTIC", "INTERNATIONAL"]

    for year in years:
        for month in months:
            for flight_type in flight_types:
                for avg_fare in avg_fares:                        
                    pax_prediction = main(year, month, avg_fare,  flight_type)
                    revenue = pax_prediction * avg_fare
                    print(f"Year: {year}, Month: {month}, Avg Fare: {avg_fare}, Flight Type: {flight_type}")
                    print(f"Predicted Number of {flight_type} Passengers: {pax_prediction:.0f}")
                    print(f"Profit: {profit_value:.2f}")

                    # Calculate VC
                    VC_value = VC(count_of_flights(pax_prediction, plane_capacity), cost_per_flight)
                    # Calculate cost
                    total_cost_value = total_cost(VC_value, FC)
                    # Calculate profit
                    profit_value = profit(revenue, total_cost_value)

