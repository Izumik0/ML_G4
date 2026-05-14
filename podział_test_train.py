import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('old_classy_datasets/Dataset_4f/clustering_results_4c.csv')

train_data, test_data = train_test_split(
    df,
    test_size=0.2,
    random_state=78,
    stratify=df['cluster_results_KMedoids']
)

print("Original Cluster Counts:")
print(df['cluster_results_KMedoids'].value_counts(normalize=True).round(3))

print("\nTraining Set Cluster Counts (85% of data):")
print(train_data['cluster_results_KMedoids'].value_counts(normalize=True).round(3))

print("\nTest Set Cluster Counts (15% of data):")
print(test_data['cluster_results_KMedoids'].value_counts(normalize=True).round(3))

test_data.to_csv('test_set_RF_4c_1.csv', index=False)
train_data.to_csv('train_set_RF_4c_1.csv', index=False)