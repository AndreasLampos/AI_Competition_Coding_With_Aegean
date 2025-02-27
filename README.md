# Dataset Descriptions

## 1. Hackthon A3 Dataset (dummy)

**Purpose:**  
This dataset is the primary source of historical flight information. It simulates "real-world" data capturing passenger volumes and aircraft capacities, which are critical for forecasting models.

**Contents & Structure:**  
- **Flight Details:** Contains records of flight dates, routes, and identifiers that link flights to specific periods and routes.
- **Capacity Information:** Provides the maximum seating or capacity figures per flight, essential for calculating load factors (LF).
- **Passenger Counts:** Includes the number of passengers that actually flew, enabling the computation of historical load factors.

**Usage in the Challenge:**  
- **Model Training:** The dataset is used to train AI models to recognize patterns and seasonal trends in passenger demand.
- **Performance Metrics:** By comparing historical capacity and passenger numbers, models can better forecast future demand and optimize load factors.
- **Seasonality & Trends:** This dataset lays the groundwork for understanding how seasonal fluctuations affect passenger numbers, allowing for adjustments in the forecasting algorithm.

**Additional Notes:**  
- Although labeled as “dummy,” the dataset is structured to closely resemble the real operational data Aegean Airlines might use.
- It serves as a critical building block in the AI in Aviation challenge, ensuring that the forecasting model is grounded in realistic operational parameters.
- **Year**: Indicates the calendar year for which the aggregated data is recorded. This helps in trend analysis over multiple years.
- **Month**: Represents the month corresponding to the data entry. This is useful for observing seasonal patterns in flight operations and demand.
- **D/I**: Stands for "Domestic/International." This cell classifies the flights based on whether they are domestic or international, allowing for separate analysis of these two market segments.
- **Count of Flights**: Shows the total number of flights operated during that specific month and year, for the given D/I category. It reflects the flight frequency.
- **Seats**: Displays the aggregated number of available seats across all the flights for that period and category. This figure is crucial for calculating capacity.
- **LF (Load Factor)**: Represents the load factor, typically calculated as the ratio of passengers (Pax) to available seats. This metric indicates how efficiently the airline’s capacity is being utilized.
- **Pax**: Indicates the total number of passengers who traveled during that month and year in the corresponding D/I category.
- **Avg. Fare**: Displays the average fare price for the flights in that period, providing insight into revenue trends and pricing strategies.

---

## 2. Hackthon Competition Dataset (dummy)

**Purpose:**  
This dataset provides competitive market insights by simulating competitors’ selling fares and other market-driven factors that might influence Aegean’s passenger demand and load factors.

**Contents & Structure:**  
- **Competitor Pricing:** Contains information on competitors’ fare prices, offering a comparative look at market pricing strategies.
- **Market Trends:** May include indicators of occupancy or promotional offers that indirectly influence passenger choices.
- **External Market Dynamics:** Helps quantify how market competition could affect Aegean's performance, acting as an external modifier to the demand forecasting model.

**Usage in the Challenge:**  
- **Complementary Analysis:** Used alongside the historical flight data to build a comprehensive forecasting model. While the A3 Dataset provides internal historical context, the Competition Dataset adds a layer of market intelligence.
- **Dynamic Forecasting:** Integrating competitor fare trends and market behavior enables the model to adjust predictions based on competitive actions, enhancing its accuracy.
- **Strategic Insights:** Assists in identifying areas where competitive pricing may influence Aegean's load factors, leading to strategic recommendations for pricing or promotional adjustments.

**Additional Notes:**  
- The dummy nature of this dataset ensures that teams focus on the modeling process without the complexities of real-world market noise.
- When combined with online sources (as outlined in the project brief), this dataset enriches the model’s inputs, offering a more robust forecasting tool. 
- **Year**: This cell records the calendar year when the data was collected. It is essential for tracking trends and changes over time in competitor behavior and market conditions.
- **Month**: This cell represents the month during which the data entry was recorded, helping to identify seasonal patterns or monthly fluctuations in pricing and capacity.
- **Carrier New**: This cell indicates the name of the competitor airline or carrier. It allows for differentiation between various competitors in the market.
- **Selling Prices**: This cell shows the fare prices offered by the competitor. It provides insight into their pricing strategies and helps in comparing market rates.
- **Capacities**: This cell records the seating or capacity information provided by the competitor. It reflects the scale of their operations and is useful for understanding their market presence relative to available service capacity.

---

## Supplementary Document: Aegean Datasets.docx

**Overview:**  
The accompanying document, **Aegean Datasets.docx**, provides broader context for the challenge. It outlines the project title, case study details, key performance metrics (passenger numbers, capacity data, competitor prices, seasonality), and suggests several online sources for real-time data. This document is crucial for understanding:
- The overall challenge goal: AI-driven passenger demand forecasting.
- How each dataset contributes to the challenge.
- The external sources that can be used to update or validate the dummy data provided in the Excel files.

---

By combining the detailed historical flight data from the **Hackthon A3 Dataset** with the competitive insights from the **Hackthon Competition Dataset**, and guided by the strategic outline in **Aegean Datasets.docx**, teams can develop a well-rounded forecasting model that maximizes load factor efficiency and provides actionable insights into market dynamics.
<br/>
<br/>
<br/>

# View our Team's Presentation with all the details, [!here](Teams presentation AI HACKATHON 2025.pptx)
