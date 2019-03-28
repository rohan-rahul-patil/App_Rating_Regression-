import pandas as pd
from sklearn import preprocessing
import numpy as np

dataset = pd.read_csv('googleplaystore.csv')
dataset2 = pd.read_csv('googleplaystore_user_reviews.csv')
dataset2["Sentiment_Polarity"] = dataset2["Sentiment_Polarity"].fillna(0)
dataset2["Sentiment_Subjectivity"] = dataset2["Sentiment_Subjectivity"].fillna(0)

aug_dataset = dataset2.groupby(['App'], as_index=False).agg({"Sentiment_Polarity": "mean", "Sentiment_Subjectivity": "mean"})

#Keep only entries which are present in both datasets
common_apps_stats = pd.merge(aug_dataset, dataset[["App"]].drop_duplicates(), how='inner')

common_dataset = pd.merge(common_apps_stats, dataset.drop_duplicates(), how='inner')
#Since there are multiple entries in the dataset1, we simply take the first entry
common_dataset = common_dataset.groupby(by="App").agg('first')