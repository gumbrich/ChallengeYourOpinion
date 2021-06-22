# project: "Challenge your opinion"
# date: June 21, 2021
# author: Johannes Knörzer

# LIBRARIES
import pandas as pd
import requests
import nltk
import schedule
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import date

tokenizer = RegexpTokenizer('\w+')
today = date.today()

# Import German stopwords
stopwords_file = open("stopwords.txt", "r")
stopwords = stopwords_file.read().split('\n')
stopwords_file.close()

########################################################################
# Case study: NZZ
def distr_nzz():


	nzz_df = pd.DataFrame(columns=['Teaser'])
	nzz_file = Path("nzz_scrape.csv")
	nzz_new_record = True
	if nzz_file.exists():
		nzz_df = pd.read_csv('nzz_scrape.csv')
		nzz_new_record = False
	nzz_words = []

	for nzz_headline_db in nzz_df['Teaser']:
		tokens = tokenizer.tokenize(nzz_headline_db)
		for word in tokens:
			if word.lower() not in stopwords:
				nzz_words.append(word)

	nzz_distr = nltk.FreqDist(nzz_words)
	
	return nzz_distr
########################################################################


########################################################################
# Case study: Spiegel
def distr_spiegel():
	
	spiegel = requests.get('http://www.spiegel.de')
	cover_spiegel = spiegel.content
	soup_spiegel = BeautifulSoup(cover_spiegel, 'html.parser')
	news_spiegel = soup_spiegel.find_all('article')
	len_spiegel = len(news_spiegel)
	spiegel_df = pd.DataFrame(columns=['Teaser'])
	spiegel_file = Path("spiegel_scrape.csv")
	spiegel_new_record = True
	if spiegel_file.exists():
		spiegel_df = pd.read_csv('spiegel_scrape.csv')
		spiegel_new_record = False
	spiegel_words = []

	for i in range(len_spiegel):
		spiegel_headline = news_spiegel[i].get('aria-label')
		
		if spiegel_headline and spiegel_headline not in spiegel_df.values:
			spiegel_df = spiegel_df.append({'Teaser': spiegel_headline, 'Date': today.strftime("%Y-%m-%d")}, ignore_index=True)

	for spiegel_headline_db in spiegel_df['Teaser']:
		tokens = tokenizer.tokenize(spiegel_headline_db)
		for word in tokens:
			if word.lower() not in stopwords:
				spiegel_words.append(word)

	spiegel_distr = nltk.FreqDist(spiegel_words)

########################################################################

########################################################################
# Case study: TAZ
def distr_taz():
	
	taz = requests.get('http://www.taz.de')
	cover_taz = taz.content
	soup_taz = BeautifulSoup(cover_taz, 'html.parser')
	news_taz = soup_taz.find_all('h3')
	len_taz = len(news_taz)
	taz_df = pd.DataFrame(columns=['Teaser'])
	taz_file = Path("taz_scrape.csv")
	taz_new_record = True
	if taz_file.exists():
		taz_df = pd.read_csv('taz_scrape.csv')
		taz_new_record = False
	taz_words = []

	for i in range(len_taz):
		taz_headline = news_taz[i].get_text()
		
		if taz_headline and taz_headline not in taz_df.values:
			taz_df = taz_df.append({'Teaser': taz_headline, 'Date': today.strftime("%Y-%m-%d")}, ignore_index=True)

	for taz_headline_db in taz_df['Teaser']:
		tokens = tokenizer.tokenize(taz_headline_db)
		for word in tokens:
			if word.lower() not in stopwords:
				taz_words.append(word)

	taz_distr = nltk.FreqDist(taz_words)

########################################################################
		
########################################################################
# Case study: Focus
def scrape_focus():
	
	focus1 = requests.get('http://www.focus.de')
	cover_focus1 = focus1.content
	soup_focus1 = BeautifulSoup(cover_focus1, 'html.parser')
	news_focus1 = soup_focus1.find_all('a')
	len_focus1 = len(news_focus1)
	focus1_df = pd.DataFrame(columns=['Teaser'])
	focus1_file = Path("focus_scrape.csv")
	focus1_new_record = True
	if focus1_file.exists():
		focus1_df = pd.read_csv('focus_scrape.csv')
		focus1_new_record = False
	focus1_words = []

	for i in range(len_focus1):
		focus1_headline = news_focus1[i].get('title')
		
		if focus1_headline and focus1_headline not in focus1_df.values:
			focus1_df = focus1_df.append({'Teaser': focus1_headline, 'Date': today.strftime("%Y-%m-%d")}, ignore_index=True)

	for focus1_headline_db in focus1_df['Teaser']:
		tokens = tokenizer.tokenize(focus1_headline_db)
		for word in tokens:
			if word.lower() not in stopwords:
				focus1_words.append(word)

	focus1_distr = nltk.FreqDist(focus1_words)

########################################################################

nzz_distr = distr_nzz()
spiegel_distr = distr_spiegel()
taz_distr = distr_taz()

# Plot frequencies
nzz_distr.plot(25, cumulative=False)
