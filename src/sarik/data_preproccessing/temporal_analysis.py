import pandas as pd 
import matplotlib.pyplot as plt
import os 
os.getcwd()
os.chdir('D:/spring 23/724-DataAnalytics_SocialMedia/Project/github/NLP_on_Social_Media_Data/src/sarik/data_preproccessing')
os.getcwd()

combined_df = pd.read_csv("data_for_visualization_.2023-03-16_12.31.29.527561.csv", sep=",", index_col=0)


combined_df['created_at'] = pd.to_datetime(combined_df['created_at'])

combined_df['hour'] = combined_df['created_at'].dt.hour
combined_df['day'] = combined_df['created_at'].dt.day_name()
combined_df['day'].unique()
total_retweets = combined_df.groupby('day')['public_metrics.retweet_count'].sum() #sum of retweets
mean_retweets = combined_df.groupby('day')['public_metrics.retweet_count'].mean() #mean of retweets
mean_retweets

# day
# Friday        148.886486
# Monday        484.246914
# Saturday      389.771739
# Sunday        340.476190
# Thursday      565.155128
# Tuesday      4882.226999
# Wednesday    1906.100220
# Name: public_metrics.retweet_count, dtype: float64

#REORDERING
total_retweets = total_retweets.reindex(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
total_retweets

#Display with matplotlib
total_retweets.plot(kind='bar', figsize=(12,8), stacked=True)
plt.xlabel('')
plt.ylabel('Total Retweets')
plt.show()
