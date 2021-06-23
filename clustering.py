# project: Challenge Your Opinion
# K-means clustering algorithm, to group news articles & visualize categories
# date: June 23, 2021

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

# ~ tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')
tokenizer = RegexpTokenizer('\w+')
p_stemmer = PorterStemmer()

# Import German stopwords
stopwords_file = open("stopwords.txt", "r")
stopwords = stopwords_file.read().split('\n')
stopwords_file.close()

########################################################################
nzz_df = pd.read_csv('nzz_scrape.csv')
spiegel_df = pd.read_csv('spiegel_scrape.csv')
taz_df = pd.read_csv('spiegel_scrape.csv')

spiegel_headlines = spiegel_df['Teaser'].values

texts_nzz = list()
texts_spiegel = list()

for spiegel_headline_db in spiegel_df['Teaser']:
	tokens_spiegel = tokenizer.tokenize(spiegel_headline_db)
	stopped_tokens_spiegel = [i for i in tokens_spiegel if not i.lower() in stopwords]
	text_spiegel = [p_stemmer.stem(t) for t in stopped_tokens_spiegel]
	texts_spiegel.append(text_spiegel)

flat_spiegel = [item for sublist in texts_spiegel for item in sublist]

vectorizer = TfidfVectorizer(max_features=20)
X = vectorizer.fit_transform(flat_spiegel)
word_features = vectorizer.get_feature_names()
print(len(word_features))

# within-cluster sum of squares
WCSS = []

n_clusters_max = 15

for i in range(1,n_clusters_max+1):
	kmeans = KMeans(n_clusters = i, init='k-means++', max_iter=500, n_init=10, random_state=0)
	kmeans.fit(X)
	WCSS.append(kmeans.inertia_)
	print(i)

plt.plot(range(1,n_clusters_max+1), WCSS)
plt.title('Elbow method')
plt.xlabel('# of clusters')
plt.ylabel('WCSS')
plt.savefig('cluster_elbow_c%i.png' % (n_clusters_max))

# clustering
kmeans = KMeans(n_clusters = 4, n_init = 40)
kmeans.fit(X)

common_words = kmeans.cluster_centers_.argsort()[:,-1:-5:-1]
for num, centroid in enumerate(common_words):
	print(str(num) + ' : ' + ', '.join(word_features[word] for word in centroid))
