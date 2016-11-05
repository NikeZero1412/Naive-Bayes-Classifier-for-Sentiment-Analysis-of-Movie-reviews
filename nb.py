# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 12:17:09 2016

@author: HOME
"""

from __future__ import division
import math
import os
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder

from collections import defaultdict


# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

# Path to dataset
PATH_TO_DATA = "D:\\Learning\\Nlp\\HW3\\large_movie_review_dataset"

TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")


def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return bow
    
def better_tokenizer(doc):
    
    
    stopset = set(stopwords.words('english'))
    bigram_measures = BigramAssocMeasures()
    bow = defaultdict(float)
    tokens = nltk.word_tokenize(doc.decode('utf8'))
    
    #include bigram collocations
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(bigram_measures.chi_sq, 500)

    for bigram_tuple in bigrams:
        x = "%s %s" % bigram_tuple
        tokens.append(x)
    #filter out stopwords
    tokens=[w for w in tokens if not w in stopset]
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return bow

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }


    def train_model(self, num_docs=None):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.

        num_docs: set this to e.g. 10 to train on only 10 docs from each category.
        """

        if num_docs is not None:
            print "Limiting to only %s docs per class" % num_docs

        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        print "Starting training with paths %s and %s" % (pos_path, neg_path)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print "REPORTING CORPUS STATISTICS"
        print "NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL]
        print "NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL]
        print "NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL]
        print "NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL]
        print "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab)

    def update_model(self, bow, label):
        """
        IMPLEMENT ME!

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """
        self.class_total_doc_counts[label]+=1
            ## For classes of words
        for k,v in bow.items():
            # for number of words we do +=1 and number of tokens we do +=v
            self.class_total_word_counts[label]+=v
            self.class_word_counts[label][k]+=v
            self.vocab.add(k)
        
        
        


    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the   tokens!
        """

        bow = tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """

        Returns the most frequent n tokens for documents with class 'label'.
        """
        return sorted(self.class_word_counts[label].items(), key=lambda (w,c): -c)[:n]

    def p_word_given_label(self, word, label):
        """
        Implement me!

        Returns the probability of word given label (i.e., P(word|label))
        according to this NB model.
        """
        count_word=self.class_word_counts[label][word]
        total_class_wordcount = self.class_total_word_counts[label]
        prob_wgl= count_word/total_class_wordcount
        return prob_wgl

    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        """
        Implement me!

        Returns the probability of word given label wrt psuedo counts.
        alpha - psuedocount parameter
        """
        V=len(self.vocab)
        total_class_wordcount = self.class_total_word_counts[label]
        count_word=self.class_word_counts[label][word]
        prob_wgl= (count_word+alpha)/(total_class_wordcount+(V*alpha))
        return prob_wgl
        

    def log_likelihood(self, bow, label, alpha):
        """
        Computes the log likelihood of a set of words give a label and psuedocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; psuedocount parameter
        """
        likelihood_log=0
        for k in bow.keys():
            p_ofwordgivenclass = self.p_word_given_label_and_psuedocount(k,label,alpha)
            likelihood_log+= math.log(p_ofwordgivenclass)
            
        return likelihood_log

    def log_prior(self, label):
        """
        Implement me!

        Returns a float representing the fraction of training documents
        that are of class 'label'.
        """
        N=self.class_total_doc_counts[POS_LABEL]+self.class_total_doc_counts[NEG_LABEL]
        class_count=self.class_total_doc_counts[label]
        prior_log=math.log(class_count/N)
        
        return prior_log

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Implement me!

        alpha - psuedocount parameter
        bow - a bag of words (i.e., a tokenized document)
        Computes the unnormalized log posterior (of doc being of class 'label').
        """
        
        return (self.log_prior(label)+self.log_likelihood(bow, label, alpha))

    def classify(self, bow, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        bow - a bag of words (i.e., a tokenized document)

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior).
        """
       
        pos_value= self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        
        neg_value= self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)
        
        
        if max(pos_value,neg_value)==neg_value:
            return NEG_LABEL
            
        else:
            return POS_LABEL
           


    def likelihood_ratio(self, word, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        numerator = 0
        denominator = 0
        likelihood_ratio = 0
        numerator = self.p_word_given_label_and_psuedocount(word, POS_LABEL, alpha)
        denominator = self.p_word_given_label_and_psuedocount(word, NEG_LABEL, alpha)
        likelihood_ratio = numerator/denominator
        return likelihood_ratio
        
        

    def evaluate_classifier_accuracy(self, alpha,num_docs=None):
        """
        Implement me!

        alpha - psuedocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        pos_path = os.path.join(TEST_DIR, POS_LABEL)
        neg_path = os.path.join(TEST_DIR, NEG_LABEL)
        true_pos=0
        true_neg=0
        false_pos=0
        false_neg=0
        post_onepos=1
        post_oneneg=1
        total=0
        accuracy=0
        if num_docs is not None:
            print "Limiting to only %s docs per class" % num_docs

        print "Starting training with paths %s and %s" % (pos_path, neg_path)
        for (p, label) in [(pos_path, POS_LABEL)]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    total+=1
                    bow=tokenize_doc(content)
                    if self.classify(bow, alpha)==label:
                        true_pos+=1
                    else:
                        false_neg+=1
                        if post_onepos==1:
                            post_onepos=0
                            print "\n Assigned label is : Negative \n"
                            print "Actual label is : ",label
                            print "Content labelled falsely negative is : \n\n",content
        for (p, label) in [(neg_path, NEG_LABEL)]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    total+=1
                    bow=tokenize_doc(content)
                    if self.classify(bow, alpha)==label:
                        true_neg+=1
                    else:
                        false_pos+=1
                        if post_oneneg==1:
                            post_oneneg=0
                            print "\n Assigned label is : Positive \n"
                            print "Actual label is : ",label
                            print "Content labelled falsely positive is : \n\n",content
        print "True Positives : ",true_pos
        print "True Negatives : ",true_neg
        print "False Positives : ",false_pos
        print "False Negatives : ",false_neg
         
        accuracy= (true_pos+true_neg)/total
        print "Accuracy is : " + str(accuracy*100) + "%"
        return accuracy
        
############## Bonus ############################################
#################################################################        
    def bonus_tokenize_and_update_model(self, doc, label):
        bow = better_tokenizer(doc)
        self.update_model(bow, label)
        


    def bonus_train_model(self, alpha, num_docs=None):
        if num_docs is not None:
             print "Limiting to only %s docs per class" % num_docs
        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        print "Starting training with paths %s and %s" % (pos_path, neg_path)
        print "Running ....."
        ###Cross validation to reduce overfitting
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            # Train 80% of the data set
            if num_docs is not None: filenames = filenames[:int(len(filenames)*0.8)]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    self.bonus_tokenize_and_update_model(content, label)
        self.report_statistics_after_training()
        print "Cross validation testing on remaining docs in training set"
        # Test 20% of Training set 
        testfiles=filenames[int(len(filenames)*0.8):]
        total_testfiles=0
        count=0
        for f in testfiles:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    total_testfiles+=1
                    bow=better_tokenizer(content)
                    if self.classify(bow, alpha)==label:
                        count+=1
        testfiles_accuracy=count/total_testfiles
        print "Cross validation accuracy on training set is : ",testfiles_accuracy
        return testfiles_accuracy

    def bonus_classifier_accuracy(self, alpha,num_docs=None):
        
        pos_path = os.path.join(TEST_DIR, POS_LABEL)
        neg_path = os.path.join(TEST_DIR, NEG_LABEL)
        true_pos=0
        true_neg=0
        false_pos=0
        false_neg=0
        
        total=0
        accuracy=0
        if num_docs is not None:
            print "Limiting to only %s docs per class" % num_docs
        print "So far so good."    
        print "Starting training with paths %s and %s" % (pos_path, neg_path)
        for (p, label) in [(pos_path, POS_LABEL)]:
            filenames = os.listdir(p)
            print "Testing positive set"
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    total+=1
                    bow=better_tokenizer(content)
                    if self.classify(bow, alpha)==label:
                        true_pos+=1
                    else:
                        false_neg+=1
        for (p, label) in [(neg_path, NEG_LABEL)]:
            filenames = os.listdir(p)
            print "Testing negative set"
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    total+=1
                    bow=better_tokenizer(content)
                    if self.classify(bow, alpha)==label:
                        true_neg+=1
                    else:
                        false_pos+=1
        print "True Positives : ",true_pos
        print "True Negatives : ",true_neg
        print "False Positives : ",false_pos
        print "False Negatives : ",false_neg
        
        accuracy= (true_pos+true_neg)/total
        print "Accuracy is : " + str(accuracy*100) + "%"
        return accuracy
############## End of  Bonus ####################################
#################################################################           
        

def produce_hw1_results():
    # PRELIMINARIES

    # QUESTION 1.1
    # uncomment the next two lines when ready to answer question 1.2
    #print "VOCABULARY SIZE: " + str(len(nb.vocab))
    #print ''

    # QUESTION 1.2
    # uncomment the next set of lines when ready to answer qeuestion 1.2
    # print "TOP 10 WORDS FOR CLASS " + POS_LABEL + " :"
    # for tok, count in nb.top_n(POS_LABEL, 10):
    #     print '', tok, count
    # print ''

    # print "TOP 10 WORDS FOR CLASS " + NEG_LABEL + " :"
    # for tok, count in nb.top_n(NEG_LABEL, 10):
    #     print '', tok, count
    # print ''
    
    
    psuedocounts=[1,2,3,4,5,6,7,8,9,10]
    accuracies=[]
    for i in range(1,len(psuedocounts)+1):
        accuracies.append(nb.evaluate_classifier_accuracy(i,num_docs=None))
    plot_psuedocount_vs_accuracy(psuedocounts,accuracies)
    
    print '[done.]'
    
def bonus_results():
    trainfiles_accuracy=nb.bonus_train_model(1,num_docs=None)
    testfiles_accuracy=nb.bonus_classifier_accuracy(1,num_docs=None)
    print "Accuracy for bonus question with better tokenizer is : ",(trainfiles_accuracy+testfiles_accuracy)*100/2
    
    

def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuries. You may want to modify this function
    to enhance your plot.
    """

    import matplotlib.pyplot as plt
    
    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.show()
    
    
if __name__ == '__main__':
    nb = NaiveBayes()
    #nb.train_model()
    nb.train_model(num_docs=None)
    #nb.evaluate_classifier_accuracy(1,num_docs=None)
    produce_hw1_results()
    #bonus_results()
    
    

