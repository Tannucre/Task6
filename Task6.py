Python 3.10.7 (tags/v3.10.7:6cc6b13, Sep  5 2022, 14:08:36) [MSC v.1933 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> import seaborn as sns
>>> data = pd.read_csv('C:\\Users\\Pc\\Desktop\\tannu\\disney_plus_titles.csv')   #Load the dataset
>>> data.head()
  show_id  ...                                        description
0      s1  ...  Two Pixar filmmakers strive to bring their uni...
1      s2  ...  The puppies go on a spooky adventure through a...
2      s3  ...  Hazel and Gus share a love that sweeps them on...
3      s4  ...  Matt Beisner uses unique approaches to modifyi...
4      s5  ...  Spidey teams up with pals to become The Spidey...

[5 rows x 12 columns]
>>> data.shape
(1368, 12)
>>> data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1368 entries, 0 to 1367
Data columns (total 12 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   show_id       1368 non-null   object
 1   type          1368 non-null   object
 2   title         1368 non-null   object
 3   director      928 non-null    object
 4   cast          1194 non-null   object
 5   country       1193 non-null   object
 6   date_added    1365 non-null   object
 7   release_year  1368 non-null   int64 
 8   rating        1366 non-null   object
 9   duration      1368 non-null   object
 10  listed_in     1368 non-null   object
 11  description   1368 non-null   object
dtypes: int64(1), object(11)
memory usage: 128.4+ KB
>>> 
>>> 
>>> data['date_added'] = pd.to_datetime(data['date_added'])
>>> data.set_index('date_added', inplace=True)
>>> month = data['title'].resample('M').count()
#plotting the time series data
plt.figure(figsize=(14, 8))
<Figure size 1400x800 with 0 Axes>
sns.lineplot(data=month)
<AxesSubplot: xlabel='date_added', ylabel='title'>
plt.title('Count the Titles added Monthly')
Text(0.5, 1.0, 'Count the Titles added Monthly')
plt.xlabel('Date')
Text(0.5, 0, 'Date')
plt.ylabel('Titles Added')
Text(0, 0.5, 'Titles Added')
plt.show()


from textblob import TextBlob
#sentiment analysis on unstructured data
data['sentiment'] = data['description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
print(data[['title', 'description', 'sentiment']].head())
                                     title  ... sentiment
date_added                                  ...          
2021-09-24                   A Spark Story  ...     0.000
2021-09-24                  Spooky Buddies  ...     0.000
2021-09-24          The Fault in Our Stars  ...     0.650
2021-09-22                 Dog: Impossible  ...     0.375
2021-09-22  Spidey And His Amazing Friends  ...     0.000

[5 rows x 3 columns]
#plotting the sentiment analysis data
plt.figure(figsize=(20, 12))
<Figure size 2000x1200 with 0 Axes>
sns.histplot(data['sentiment'], kde=True)
<AxesSubplot: xlabel='sentiment', ylabel='Count'>
plt.title('Sentiment Distribution')
Text(0.5, 1.0, 'Sentiment Distribution')
plt.xlabel('Sentiment')
Text(0.5, 0, 'Sentiment')
plt.ylabel('Frequency')
Text(0, 0.5, 'Frequency')
plt.show()


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('C:\\Users\\Pc\\Desktop\\tannu\\disney_plus_titles.csv')
data['sentiment'] = data['description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
x = data[['sentiment']]    #selecting features for clustering
scaler = StandardScaler()
KeyboardInterrupt
x_scaler = scaler.fit_transform(x)
# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=35)
data['cluster'] = kmeans.fit_predict(x_scaler)
# Plot the clusters
plt.figure(figsize=(12, 4))
<Figure size 1200x400 with 0 Axes>
sns.scatterplot(x=data.index, y='sentiment', hue='cluster', data=data)
<AxesSubplot: xlabel='None', ylabel='sentiment'>
plt.title('Clustering of Titles Based on Sentiment')
Text(0.5, 1.0, 'Clustering of Titles Based on Sentiment')
plt.xlabel('Index')
Text(0.5, 0, 'Index')
plt.ylabel('Sentiment')
Text(0, 0.5, 'Sentiment')
plt.legend(title='Cluster')
<matplotlib.legend.Legend object at 0x00000224E5965600>
plt.show()





