import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the merged dataset
df = pd.read_csv('merged_set.csv')

# Calculate correlation matrix for D and I variables
correlation_matrix_DI = df[['pax_D', 'seats_D', 'flights_D', 'avg_fare_D', 'LF_D', 'pax_I', 'seats_I', 'flights_I', 'avg_fare_I', 'LF_I']].corr()

# Create a figure for D and I variables correlation matrix
plt.figure(figsize=(10, 8))

# Create heatmap using seaborn for D and I variables
sns.heatmap(correlation_matrix_DI, 
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            vmin=-1, vmax=1,  # Set correlation range
            center=0,  # Center the colormap at 0
            fmt='.2f',  # Format correlation values to 2 decimal places
            square=True,  # Make the plot square-shaped
            annot_kws={'size': 8})  # Make annotation text smaller

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Add a title
plt.title('Correlation Matrix Heatmap (D and I Variables)', pad=10, fontsize=12)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with higher DPI for better quality
plt.savefig('correlation_heatmap_DI.png', dpi=300, bbox_inches='tight')

# Display the correlation matrix as a table
print("\nCorrelation Matrix (D and I Variables):")
print(correlation_matrix_DI.round(2))

# Save correlation matrix to CSV
correlation_matrix_DI.round(2).to_csv('correlation_matrix_DI.csv')

plt.show()

# Calculate correlation matrix for selected variables
correlation_matrix_selected = df[['pax_D', 'pax_I', 'avg_fare_D', 'avg_fare_I', 'Selling Prices ', 'Capacities ', 'Month_Rank']].corr()

# Create a figure for selected variables correlation matrix
plt.figure(figsize=(10, 8))

# Create heatmap using seaborn for selected variables
sns.heatmap(correlation_matrix_selected, 
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            vmin=-1, vmax=1,  # Set correlation range
            center=0,  # Center the colormap at 0
            fmt='.2f',  # Format correlation values to 2 decimal places
            square=True,  # Make the plot square-shaped
            annot_kws={'size': 8})  # Make annotation text smaller

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Add a title
plt.title('Correlation Matrix Heatmap (Selected Variables)', pad=10, fontsize=12)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with higher DPI for better quality
plt.savefig('correlation_heatmap_selected.png', dpi=300, bbox_inches='tight')

# Display the correlation matrix as a table
print("\nCorrelation Matrix (Selected Variables):")
print(correlation_matrix_selected.round(2))

# Save correlation matrix to CSV
correlation_matrix_selected.round(2).to_csv('correlation_matrix_selected.csv')

plt.show()