import json
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from textblob import TextBlob, Word
from textblob.en import Spelling

class queryActions:

	def stopword_set(self):
		"""
		Adds additional stopwords to the NLTK stopword set

		Returns
		-------
		A set of stopwords

		"""
		stopword = set(stopwords.words('english'))
		stopword.add('viz')
		alpha = set(string.ascii_lowercase)
		return set.union(stopword,alpha)

	# Training on Cranfield Dataset
	def train_cranfield(self, data):
		"""
		Function to train spell checker specifically for Cranfield dataset
		
		Parameters
		----------
		arg1 : list
			List of dictonaries containing the documents

		Returns
		-------
		None

		Creates a train.txt file in the working directory. Needs to be run only once.
		"""
		n = len(data)
		stop_words = stopword_set()
		text = ""
		for i in range(n):
			blob = TextBlob(data[i]['body'])
			filtered_body = [w for w in blob.words if not w in stop_words]
			lemma_body = [w.lemmatize() for w in filtered_body]
			filtered_body_text = ' '.join(lemma_body)
			text = text + " " + filtered_body_text

		spelling = Spelling(path = pathToFile)
		print("training")
		spelling.train(text, pathToFile)


	def spell_check_query(self, query, pathToFile):
		"""
		Function to correct spelling errors in query based on words present in Cranfield dataset.
		
		Parameters
		----------
		arg1 : list
			A list containing tokenized and preprocessed words in query.

		Returns
		-------
		A string of corrected query.

		"""

		spelling_trained = Spelling(path = pathToFile)
		corrected_query = ""
		for word in query:
			corrected_query = corrected_query + " " + spelling_trained.suggest(word)[0][0]
		
		return corrected_query

	def digit_removal(self, query):
		"""
		Removes digits from queries.

		Parameters
		----------
		arg1 : string
			Query which is passed as a string.

		Returns
		--------
		string
			Query after removing numbers.
		"""
		return re.sub(r'[0-9]','',query)


	def query_tokenizer(self, query):
		"""
		Function to tokenize query using Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : string
			Query passed as a string

		Returns
		-------
		list
			List containing sequence of tokens of the query
		"""
		return TreebankWordTokenizer().tokenize(query)


	def query_lemmatizer(self, query):
		"""
		Function to lemmatize tokens in query

		Parameters
		----------
		arg1 : list
			A list containing a sequence of tokens

		Returns
		-------
		list
			List containing lemmatized tokens of the query
		"""

		lemmatized_query = [WordNetLemmatizer().lemmatize(w) for w in query]

		return lemmatized_query


	def query_stopword(self, query):
		"""
		Function to remove stop words from query using a modified set of stopwords

		Parameters
		----------
		arg1 : list
			List containing a sequence of tokens

		Returns
		-------
		list
			List containing sequence of tokens after stopword removal
		"""
		
		stopwordset = self.stopword_set()
		filtered_query = [w for w in query if not w in stopwordset]

		return filtered_query
