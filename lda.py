# project: "Challenge your opinion", LDA analysis
# date: June 19, 2021
# author: Johannes Knörzer

# LIBRARIES
import pandas as pd
import nltk
import gensim
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim.models import LdaMulticore
from matplotlib.ticker import FuncFormatter

tokenizer = RegexpTokenizer('\w+')
p_stemmer = PorterStemmer()

# Import German stopwords
stopwords_file = open("stopwords.txt", "r")
stopwords = stopwords_file.read().split('\n')
stopwords_file.close()

########################################################################
# Categorizing Sentences
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)
########################################################################

########################################################################
# Find topic ID with largest contribution to a given headline
def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get dominant topic, contribution (in percent) and keywords for each headline
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
########################################################################


########################################################################
nzz_df = pd.read_csv('nzz_scrape.csv')
spiegel_df = pd.read_csv('spiegel_scrape.csv')
taz_df = pd.read_csv('spiegel_scrape.csv')

texts_nzz = list()
texts_spiegel = list()

########################################################################
for nzz_headline_db in nzz_df['Teaser']:
	tokens_nzz = tokenizer.tokenize(nzz_headline_db)
	stopped_tokens_nzz = [i for i in tokens_nzz if not i.lower() in stopwords]
	text_nzz = [p_stemmer.stem(t) for t in stopped_tokens_nzz]
	texts_nzz.append(text_nzz)
for spiegel_headline_db in spiegel_df['Teaser']:
	tokens_spiegel = tokenizer.tokenize(spiegel_headline_db)
	stopped_tokens_spiegel = [i for i in tokens_spiegel if not i.lower() in stopwords]
	text_spiegel = [p_stemmer.stem(t) for t in stopped_tokens_spiegel]
	texts_spiegel.append(text_spiegel)

print(stopped_tokens_spiegel)

print(f'Einträge aus NZZ: {len(texts_nzz)}')
print(f'Einträge aus Spiegel: {len(texts_spiegel)}')

########################################################################
# LDA analysis

n_topics = 3

########################################################################
dictionary_nzz = corpora.Dictionary(texts_nzz)
corpus_nzz = [dictionary_nzz.doc2bow(text) for text in texts_nzz]
ldamodel_nzz = gensim.models.ldamodel.LdaModel(corpus_nzz, num_topics=n_topics, id2word = dictionary_nzz, passes=100, minimum_probability=0.01, alpha='asymmetric', eta=0.01)
########################################################################
dictionary_spiegel = corpora.Dictionary(texts_spiegel)
corpus_spiegel = [dictionary_spiegel.doc2bow(text) for text in texts_spiegel]
ldamodel_spiegel = gensim.models.ldamodel.LdaModel(corpus_spiegel, num_topics=n_topics, id2word = dictionary_spiegel, passes=100, minimum_probability=0, alpha='asymmetric')

mdiff, annotation = ldamodel_nzz.diff(ldamodel_spiegel, distance='jaccard', num_words=50) # try Hellinger distance

min_indices = np.unravel_index(mdiff.argmin(), mdiff.shape)
i_min, j_min = min_indices[0], min_indices[1]

# Plot Jaccard distance in matrix form
# ~ plt.imshow(mdiff)
# ~ plt.colorbar()
# ~ plt.show()

########################################################################
# Test case study for comparing NZZ and Spiegel Data:
# pick most relevant headlines and find intersections

df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel_nzz, corpus=corpus_nzz, texts=texts_nzz)
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic = df_dominant_topic.sort_values('Topic_Perc_Contrib', ascending=False)
topic_1_nzz, topic_1_spiegel = [], []
for i in range(50):
	if int(df_dominant_topic['Dominant_Topic'].values[i]) == i_min:
		topic_1_nzz.append(nzz_df['Teaser'][df_dominant_topic['Document_No'].values[i]])
# ~ df_dominant_topic.to_csv('test1.csv', index=False)

df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel_spiegel, corpus=corpus_spiegel, texts=texts_spiegel)
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic = df_dominant_topic.sort_values('Topic_Perc_Contrib', ascending=False)
for i in range(50):
	if int(df_dominant_topic['Dominant_Topic'].values[i]) == j_min:
		topic_1_spiegel.append(spiegel_df['Teaser'][df_dominant_topic['Document_No'].values[i]])
# ~ df_dominant_topic.to_csv('test2.csv', index=False)

escapes = '\b\n\r\t\\'
for i in range(3):
	for c in escapes: 
		topic_1_nzz[i] = topic_1_nzz[i].replace(c, '') 
		topic_1_spiegel[i] = topic_1_spiegel[i].replace(c, '') 

print("NZZ headlines:")
print(topic_1_nzz[0:3])
print("Spiegel headlines:")
print(topic_1_spiegel[0:3])


########################################################################
# Plot a chart diagram of major topics in news outlets

# Assign top n keywords to each topic
top_n = 2

############################
# (1) NZZ
dominant_topics_nzz, topic_percentages_nzz = topics_per_document(model=ldamodel_nzz, corpus=corpus_nzz, end=-1)
df = pd.DataFrame(dominant_topics_nzz, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_headline_nzz = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_headline_nzz = dominant_topic_in_each_headline_nzz.to_frame(name='count').reset_index()

# Topic distribution
topic_weightage_by_headline_nzz = pd.DataFrame([dict(t) for t in topic_percentages_nzz])
df_topic_weightage_by_headline_nzz = topic_weightage_by_headline_nzz.sum().to_frame(name='count').reset_index()

# Top n keywords for each topic
topic_top_n_words_nzz = [(i, topic) for i, topics in ldamodel_nzz.show_topics(formatted=False) 
                                 for j, (topic, wt) in enumerate(topics) if j < top_n]
df_top_n_words_stacked_nzz = pd.DataFrame(topic_top_n_words_nzz, columns=['topic_id', 'words'])
df_top_n_words_nzz = df_top_n_words_stacked_nzz.groupby('topic_id').agg(', \n'.join)
df_top_n_words_nzz.reset_index(level=0,inplace=True)

############################
# (2) Spiegel
dominant_topics_spiegel, topic_percentages_spiegel = topics_per_document(model=ldamodel_spiegel, corpus=corpus_spiegel, end=-1)
df = pd.DataFrame(dominant_topics_spiegel, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_headline_spiegel = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_headline_spiegel = dominant_topic_in_each_headline_spiegel.to_frame(name='count').reset_index()

# Topic distribution
topic_weightage_by_headline_spiegel = pd.DataFrame([dict(t) for t in topic_percentages_spiegel])
df_topic_weightage_by_headline_spiegel = topic_weightage_by_headline_spiegel.sum().to_frame(name='count').reset_index()

# Top n keywords for each topic
topic_top_n_words_spiegel = [(i, topic) for i, topics in ldamodel_spiegel.show_topics(formatted=False) 
                                 for j, (topic, wt) in enumerate(topics) if j < top_n]
df_top_n_words_stacked_spiegel = pd.DataFrame(topic_top_n_words_spiegel, columns=['topic_id', 'words'])
df_top_n_words_spiegel = df_top_n_words_stacked_spiegel.groupby('topic_id').agg(', \n'.join)
df_top_n_words_spiegel.reset_index(level=0,inplace=True)

############################
# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300, sharey=True)

# Topic Distribution by Dominant Topics
ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_headline_nzz, width=.5, color='firebrick')
ax1.set_xticks(range(df_dominant_topic_in_each_headline_nzz.Dominant_Topic.unique().__len__()))
tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x+1)+ '\n' + df_top_n_words_nzz.loc[df_top_n_words_nzz.topic_id==x, 'words'].values[0])
ax1.xaxis.set_major_formatter(tick_formatter)
ax1.set_title('# of headlines by dominant topic: NZZ', fontdict=dict(size=10))
ax1.set_ylabel('Number of Headlines')
ax1.set_ylim(0, 500)

ax2.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_headline_spiegel, width=.5, color='steelblue')
ax2.set_xticks(range(df_dominant_topic_in_each_headline_spiegel.Dominant_Topic.unique().__len__()))
tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x+1)+ '\n' + df_top_n_words_spiegel.loc[df_top_n_words_spiegel.topic_id==x, 'words'].values[0])
ax2.xaxis.set_major_formatter(tick_formatter)
ax2.set_title('# of headlines by dominant topic: Spiegel', fontdict=dict(size=10))
ax2.set_ylabel('Number of Headlines')
ax2.set_ylim(0, 500)

plt.tight_layout()
plt.savefig('headlines_by_topic.png', dpi=300)
