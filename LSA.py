import numpy as np
import pandas as pd
import re
import math
import json
import nltk
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics.pairwise import cosine_similarity


class LemmaTokenizer:
	
	def __init__(self):
		self.tokenize = RegexpTokenizer(r'\b\w{1,}\b')
		self.ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`', '(', ')', '+', '-', '--', '*', '/', '']

	def __call__(self, articles):
		return [WordNetLemmatizer().lemmatize(re.sub(r'[0-9]', "", t)) for t in self.tokenize.tokenize(articles) if re.sub(r'[0-9]', "", t) not in self.ignore_tokens]



class LatentSemantic:

	def __init__(self, queries, doc_ids, docs, stop_words, n_components):
		"""
		Initialized several useful variables within the class.
		"""
		self.queries = queries
		self.doc_ids = doc_ids
		self.docs = docs
		self.stop_words = stop_words
		self.tokenizer = LemmaTokenizer()
		self.token_stop = self.tokenizer(' '.join(self.stop_words))
		self.n_components = n_components

	def build(self):
		"""
		Creates tf-idf term-document matrix and performs SVD decomposition.

		Parameters
		----------
		None

		Returns
		-------
		arg1 : array
			Two dimensional array containing tf-idf score for each query vector.
		"""

		sample_tfidf = TfidfVectorizer(lowercase = True, stop_words = self.token_stop, tokenizer = self.tokenizer)
		sample_sparse = sample_tfidf.fit_transform(self.docs)
		tfidf = sample_sparse.toarray()

		self.u, self.s, self.vh = np.linalg.svd(np.transpose(tfidf), full_matrices = False)
		sample_sparse_q = sample_tfidf.transform(self.queries)
		tfidf_q = sample_sparse_q.toarray()

		return tfidf_q

	def LSI(self, tfidf_q):
		"""
		Using pre-determined number of components, computes the document and query vectors in n-dimensional latent space.

		Parameters
		----------
		arg1 : array
			Two dimensional array containing tf-idf scores for each query vector.

		Returns
		-------
		arg1 : array
			Two dimensional array containing document vectors in the new n-dimensional space.
		arg2 : array
			Two dimensional array containing query vectors in the new n-dimensional space.
		"""

		T = self.u[:,:self.n_components]
		S = np.diag(self.s[:self.n_components])
		Dt = self.vh[:self.n_components,:]
		doc_vectors = np.dot(np.transpose(Dt),S)
		query_vectors = np.dot(tfidf_q, T)

		return doc_vectors, query_vectors

	def rank(self, doc_vectors, query_vectors, docIDs):
		"""
		Ranks documents for each query

		Parameters
		----------
		arg1 : array
			Two dimensional array containing document vectors.
		arg2 : array
			Two dimensional array containing query vectors.

		Returns
		-------
		list
			List of lists where each sublist contains cosine similarity score of documents with the corresponding query
		"""
		doc_IDs_ordered = list()
		for q_vector in query_vectors:
			retrieved_docs = dict()
			key = 0
			for doc_vector in doc_vectors:
				cosine = 1 - distance.cosine(doc_vector, q_vector)
				retrieved_docs[docIDs[key]] = cosine
				key += 1
			doc_IDs_ordered.append(sorted(retrieved_docs,reverse=True,key = lambda x: retrieved_docs[x]))

		return doc_IDs_ordered

	def cosSimilar(self, doc_vectors, query_vectors):
		return cosine_similarity(np.asarray(query_vectors), np.asarray(doc_vectors))

	
	def queryNDCG(self, query_doc_IDs_ordered, query_id, qrels, k):
			"""
			Computation of nDCG of the Information Retrieval System
			at given value of k for a single query

			Parameters
			----------
			arg1 : list
				A list of integers denoting the IDs of documents in
				their predicted order of relevance to a query
			arg2 : int
				The ID of the query in question
			arg3 : list
				A list of dictionaries containing document-relevance
				judgements - Refer cran_qrels.json for the structure of each
				dictionary
			arg4 : int
				The k value

			Returns
			-------
			float
				The nDCG value as a number between 0 and 1
			"""

			nDCG = -1
			DCG = 0
			iDCG = 0
			retrieved_docs = query_doc_IDs_ordered[:k]
			true_doc_IDs = dict()
			flag = 0
			for dic in qrels:
				if(int(dic["query_num"]) == query_id):
					flag = 1
					true_doc_IDs[int(dic["id"])] = 5 - dic['position']
				elif(flag == 1):
					break
			ideal_order = sorted(query_doc_IDs_ordered, key = lambda x: true_doc_IDs[x] if x in true_doc_IDs else 0, reverse = True)[:k]
			for i in range(1,k+1):
				if retrieved_docs[i-1] in true_doc_IDs:
					DCG += true_doc_IDs[retrieved_docs[i-1]]/math.log2(i+1)
				if ideal_order[i-1] in true_doc_IDs:
					iDCG += true_doc_IDs[ideal_order[i-1]]/math.log2(i+1)

			if iDCG == 0:
				iDCG = 1	
			nDCG = DCG/iDCG

			return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
			"""
			Computation of nDCG of the Information Retrieval System
			at a given value of k, averaged over all the queries

			Parameters
			----------
			arg1 : list
				A list of lists of integers where the ith sub-list is a list of IDs
				of documents in their predicted order of relevance to the ith query
			arg2 : list
				A list of IDs of the queries for which the documents are ordered
			arg3 : list
				A list of dictionaries containing document-relevance
				judgements - Refer cran_qrels.json for the structure of each
				dictionary
			arg4 : int
				The k value

			Returns
			-------
			float
				The mean nDCG value as a number between 0 and 1
			"""
			
			meanNDCG = -1
			total_NDCG = 0
			count = 0
			for i in query_ids:
				total_NDCG = total_NDCG + self.queryNDCG(doc_IDs_ordered[count] , i , qrels, k)
				count = count+1
			if count == 0:
				meanNDCG = 0
			else:
				meanNDCG = total_NDCG/count

			return meanNDCG
