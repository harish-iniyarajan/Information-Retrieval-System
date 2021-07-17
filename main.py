import os
from queryPreprocessor import *
from LSA import *
import json
from sklearn.metrics import ndcg_score

dirname = os.path.dirname(__file__)
dataset_path = os.path.join(dirname, 'cranfield\cran_docs.json')
query_path = os.path.join(dirname, 'cranfield\cran_queries.json')
qrels_path = os.path.join(dirname, 'cranfield\cran_qrels.json')
pathToFile = os.path.join(dirname, 'train.txt')

class mySearchEngine:

    def __init__(self):
        """
        Loads necessary data and creates necessary classes
        """
        self.docs_json = json.load(open(dataset_path, 'r'))[:]
        self.doc_ids, self.docs =  [item["id"] for item in self.docs_json],[item["body"] for item in self.docs_json]
        queries_json = json.load(open(query_path, 'r'))[:]
        self.queries = [item["query"] for item in queries_json]
        self.qrels = json.load(open(qrels_path, 'r'))[:]
        self.query_ids = [item["query number"] for item in queries_json]
        self.n_components = 300
        self.query_methods = queryActions()
        self.lsa = LatentSemantic(self.queries, self.doc_ids, self.docs, self.query_methods.stopword_set(), self.n_components)

    def preprocessQuery(self, query):
        """
        Function to preprocess query

        Parameters
        ----------
        arg1 : string
            Query passed as a string

        Returns
        -------
        string
            Query as a sentence after being preprocessed and corrected.
        """
        
        drquery = self.query_methods.digit_removal(query)
        qlist = self.query_methods.query_tokenizer(query)
        lemma_qlist = self.query_methods.query_lemmatizer(qlist)
        filter_qlist = self.query_methods.query_stopword(lemma_qlist)
        corrected_query = self.query_methods.spell_check_query(filter_qlist, pathToFile)

        return corrected_query
    
    def trainSpellChecker(self):
        """
        Trains spell checker on specific dataset - in this case, Cranfield dataset
        
        Creates a train.txt file in working directory and needs to run only once.
        """

        self.query_methods.train_cranfield(json.load(open(dataset_path)))

    def customQueryRank(self, query):
        """
        Ranks documents for a custom query.

        Parameters
        ----------
        arg1 : string
            User given custom query.

        Returns
        -------
        list
            List containing document indiced of top 10 retrieved documents
        """

        cquery = self.preprocessQuery(query)
        self.lsa_custom = LatentSemantic([cquery], self.doc_ids, self.docs, self.query_methods.stopword_set(), self.n_components)
        qtfidf = self.lsa_custom.build()
        dvec, qvec = self.lsa_custom.LSI(qtfidf)
        custom_score = self.lsa_custom.rank(dvec, qvec, self.doc_ids)

        top_doc_ids = np.argsort(-np.asarray(custom_score))

        return top_doc_ids[0][0:10]

    def rankQueries(self):
        """
        Ranks documents for queries present in cran_queries

        Parameters
        ----------
        None

        Returns
        -------
        list
            List of list where each sub list contains cosine similarity of documents for that corresponding query
        """

        qtfidf1 = self.lsa.build()
        dvec1, qvec1 = self.lsa.LSI(qtfidf1)
        doc_score = self.lsa.rank(dvec1, qvec1, self.doc_ids)
        #doc_score = self.lsa.cosSimilar(dvec1, qvec1)

        return doc_score

    def calcnDCG(self):
        """
        Calculates nDCG score for queries in cran_qrels

        Parameters
        ----------
        None

        Returns
        -------
        float
            nDCG score
        """
        dscore = self.rankQueries()

        return self.lsa.meanNDCG(dscore, self.query_ids, self.qrels, len(self.docs))

if __name__ == "__main__":
    engine = mySearchEngine()
    char = input("Do you want to enter a custom query [Y/n]: ")

    if ((char == 'Y') or (char == 'y')):
        quer = input("Enter your query: ")
        d_ids = engine.customQueryRank(quer)
        print("Titles of top 10 retrieved documents are:")
        for i in d_ids:
            print(engine.docs_json[i]['title'])
        

    elif ((char == 'N') or (char == 'n')):
        print("nDCG score evaluated on documents given in cran_qrels is: ", engine.calcnDCG())
    
    else:
        print('Invalid input')    