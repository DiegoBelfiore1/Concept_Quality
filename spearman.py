import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '______.csv'

df = pd.read_csv(file_path)

concepts = np.array(df)

corr_matrix, p_values = spearmanr(concepts, axis=0)  

# Just the first ten concepts to generate the heatmap
corr_matrix_10 = corr_matrix[:10, :10]

diversity = np.mean(np.abs(corr_matrix_10))

print(f"Diversity (mean absolute Spearman correlation for top 10 concepts): {diversity:.4f}")

plt.switch_backend('Agg')

plt.figure(figsize=(8, 6))

sns.heatmap(corr_matrix_10, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title(f'Matrix of Top 10 Concepts. Spearman Correlation {diversity:.4f}')

plt.savefig('correlation_heatmap_top10.png')

print("correlation_heatmap_top10.png saved.")