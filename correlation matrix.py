mport numpy as np
import matplotlib.pyplot as plt
def correlation_matrix(data):
    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)
    
    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.xticks(range(len(corr_matrix)), range(len(corr_matrix)), rotation=90)
    plt.yticks(range(len(corr_matrix)), range(len(corr_matrix)))
    plt.show()
    
    return corr_matrix

import pandas as pd
# Example usage
if __name__ == "__main__":
    # Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
    data = pd.read_csv('your_dataset.csv')
    
    # Select the relevant columns for correlation analysis (replace with your actual column names)
    selected_data = data[['column1', 'column2', 'column3']]
    
    # Calculate and plot the correlation matrix
    corr_matrix = correlation_matrix(selected_data.values)
    print("Correlation Matrix:")
    print(corr_matrix)

    # Save the correlation matrix to a CSV file
    np.savetxt('correlation_matrix.csv', corr_matrix, delimiter=',')
    