import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the merged dataset
df = pd.read_csv('merged_set.csv')

# Calculate correlation matrix
correlation_matrix = df.drop(['Year', 'Month'], axis=1).corr()

# Create a figure with smaller size (reduced from 12,10 to 8,6)
plt.figure(figsize=(8, 6))

# Create heatmap using seaborn
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            vmin=-1, vmax=1,  # Set correlation range
            center=0,  # Center the colormap at 0
            fmt='.2f',  # Format correlation values to 2 decimal places
            square=True,  # Make the plot square-shaped
            annot_kws={'size': 6})  # Make annotation text smaller

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

# Add a title
plt.title('Correlation Matrix Heatmap', pad=10, fontsize=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with higher DPI for better quality
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

# Display the correlation matrix as a table
print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

# Save correlation matrix to CSV
correlation_matrix.round(2).to_csv('correlation_matrix.csv')

plt.show()