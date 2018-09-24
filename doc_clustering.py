
# coding: utf-8

# In[128]:


from __future__ import print_function
import pandas as pd
import numpy as np
import nltk
import os
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|[^\w\s]+')



# In[129]:


dir = "/home/varma/nikh/COLUMNS DATA SET/"

authors = []    #storing author names
titles = []     #storing author for each document
chapters = []   #contains the text from each file
doc = {}

for root, subfolders, files in os.walk(dir):
    # adding authors to list
    ti = []
    author = root.split("/")[-1]
    

    if author == "":
        continue
    authors.append(author)

    for i in files:
        # adding article names to the list
        ti.append(i)
        if (i.startswith("unk")):
            titles.append("UNKNOWN")
        else:
            titles.append(author)
        # print(root + "/" + i + "\n")
        f = codecs.open(root + "/" + i, "r", encoding="utf-8", errors='ignore')
        str = f.read()
        chapters.append(str)
        # j += 1
        # doc[authors[-1]] = list(doc[[authors[-1]]].append(i))
    doc[author] = ti

print(authors)
all_text = ' '.join(chapters)
# print(titles[:2])


ranks = []
for i in range(1, len(titles) + 1):
    ranks.append(i)


# In[203]:


def PredictAuthors(fvs):
    """
    Use k-means clustering to fit a model
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
    km = KMeans(n_clusters=7)
    km.fit(fvs)

    return km


# In[131]:


def LexicalFeatures():
    """
    Compute feature vectors for word and punctuation features
    """
    num_chapters = len(chapters)
    fvs_lexical = np.zeros((len(chapters), 3), np.float64)
    fvs_punct = np.zeros((len(chapters), 4), np.float64)
    for e, ch_text in enumerate(chapters):
        # note: the nltk.word_tokenize includes punctuation
        tokens = nltk.word_tokenize(ch_text.lower())
        words = word_tokenizer.tokenize(ch_text.lower())
        sentences = sentence_tokenizer.tokenize(ch_text)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                       for s in sentences])

        # average number of words per sentence
        fvs_lexical[e, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[e, 1] = words_per_sentence.std()
        # Lexical diversity
        fvs_lexical[e, 2] = len(vocab) / float(len(words))

        # Commas per sentence
        fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
        # Semicolons per sentence
        fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
        # Colons per sentence
        fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))
        # Period per sentence
        fvs_punct[e, 3] = tokens.count('.') / float(len(sentences))

    # apply whitening to decorrelate the features
    fvs_lexical = whiten(fvs_lexical)
    fvs_punct = whiten(fvs_punct)

    return fvs_lexical, fvs_punct



# In[132]:


def BagOfWords():
    """
    Compute the bag of words feature vectors, based on the most common words
     in the whole book
    """
    # get most common words in the whole book
    NUM_TOP_WORDS = 100
    bag_text = all_text.replace('.', '')
    bag_text = bag_text.replace(',', '')
    bag_text = bag_text.replace(':', '')
    all_tokens = nltk.word_tokenize(bag_text)
    fdist = nltk.FreqDist(all_tokens)
    #     print()
    vocab = list(nltk.FreqDist.keys(fdist))

    vocab = fdist.most_common(NUM_TOP_WORDS)
    print(vocab)
    vocab = [v[0] for v in vocab]

    # use sklearn to create the bag for words feature vector for each chapter
    vectorizer = CountVectorizer(vocabulary=vocab)
    fvs_bow = vectorizer.fit_transform(chapters).toarray().astype(np.float64)
    #     print(fvs_bow)

    # normalise by dividing each row by its Euclidean norm
    # fvs_bow /= np.c_[np.apply_along_axis(np.linalg.norm, 1, fvs_bow)]

    #     print(fvs_bo

    return np.nan_to_num(fvs_bow)


# In[133]:


def SyntacticFeatures():
    """
    Extract feature vector for part of speech frequencies
    """

    def token_to_pos(ch):
        tokens = nltk.word_tokenize(ch)
        return [p[1] for p in nltk.pos_tag(tokens)]

    chapters_pos = [token_to_pos(ch) for ch in chapters]
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    fvs_syntax = np.array([[ch.count(pos) for pos in pos_list]
                           for ch in chapters_pos]).astype(np.float64)

    # normalise by dividing each row by number of tokens in the chapter
    fvs_syntax /= np.c_[np.array([len(ch) for ch in chapters_pos])]

    #     print(fvs_syntax)

    return fvs_syntax


# In[134]:


def PlotCluster(labels, centroids, data):
    from matplotlib import pyplot
    k = 7
    for i in range(0, k):
        # select only data observations with cluster label == i
        ds = data[np.where(labels == i)]
        # plot the data observations
        pyplot.plot(ds[:, 0], ds[:, 1], 'o')
        # plot the centroids
        lines = pyplot.plot(centroids[i, 0], centroids[i, 1], 'kx')
        # make the centroid x's bigger
        pyplot.setp(lines, ms=15.0)
        pyplot.setp(lines, mew=2.0)
    pyplot.show()


# In[169]:


def score(clusters, num_clusters):
    films = {'title': titles, 'rank': ranks, 'synopses': synopses, 'cluster': clusters
    , 'authors': authors}
    frame = pd.DataFrame(films, index=[clusters], columns=['rank', 'title', 'cluster', 'author'])
    # print(frame)
    
    frame['cluster'].value_counts()
    
    grouped = frame['rank'].groupby(frame['cluster'])
    print(grouped.mean())
    
    # In[9]
    
    # print('top terms per cluster')
    
    clust_details=[]
    # order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        # print('cluster %d titles:\n' % i, end='')
        cou={}
        for title in frame.ix[i]['title']:
            cou[title] = 0
        print('cluster  length:   %d\n' % len(frame.ix[i]['title']), end='')
        for title in frame.ix[i]['title']:
            cou[title] += 1
            # print(' %s, ' % title, end='')
        # print('\n')
        # print(cou)
        p = max(cou.values())/sum(cou.values())
        r = max(cou.values())/50
        print("Precision = ", p, " Recall = ", r, "F-Score = ", ((2*p*r)/(p+r)))
        clust_details.append(cou)
        print()
    print()
    print()
    
    F = 0
    for cdict in clust_details:
        N1 = max(cdict.values())
        N2 = sum(cdict.values())
        N3 = 50
        P = N1 / N2
        R = N1 / N3
        F += (2 * P * R) / (P + R)
        
    print("BCubed F-Score = ", (F/num_clusters))
    print()
    print()



# In[206]:


if __name__ == '__main__':
    
    NUM_TOP_WORDS = 100
    bag_text = all_text.replace('.', '')
    bag_text = bag_text.replace(',', '')
    bag_text = bag_text.replace(':', '')
    all_tokens = nltk.word_tokenize(bag_text)
    fdist = nltk.FreqDist(all_tokens)
    #     print()
    vocab = list(nltk.FreqDist.keys(fdist))

    vocab = fdist.most_common(NUM_TOP_WORDS)
    print(vocab)
    vocab = [v[0] for v in vocab]

    # use sklearn to create the bag for words feature vector for each chapter
    vectorizer = CountVectorizer(vocabulary=vocab)
    fvs_bow = vectorizer.fit_transform(chapters).toarray().astype(np.float64)
    
    
    feature_set = fvs_bow
    # print(feature_set)
    results = PredictAuthors(feature_set)
    print(results.labels_)
    PlotCluster(results.labels_, results.cluster_centers_, feature_set)
    clusters = results.labels_.tolist()
    score(clusters,  7)

#     feature_sets = list(LexicalFeatures())
#     feature_sets.append(BagOfWords())
#     feature_sets.append(SyntacticFeatures())
#     # print(feature_sets)
#     
#     res = PredictAuthors(BagOfWords())
#     PlotCluster(res.labels_, res.cluster_centers_, BagOfWords())
#         
#     clu_cent =  0 * np.ones((7,2)) 
#     clu_lab = []
# 
#     classifications = [PredictAuthors(fvs) for fvs in feature_sets]
# #     print(classifications)
#     i = 0
#     feature = ""
# #     for clusters in classifications:
# #         PlotCluster(clusters)
# 
#     for results in classifications:
#         if i == 0:
#             feature = "Lexical Features: "
#         elif i == 1:
#             feature = "Punctuation Features: "
#         elif i == 2:
#             feature = "Bag of Words: "
#         elif i == 3:
#             feature = "Syntactic Features: "
#         # the first 5 are akbar for sure and we know that.
#         #        if results.labels_[0] == 0: results.labels_[0] = 1
#         #        if results.labels_[1] == 0: results.labels_[1] = 1
#         #        if results.labels_[2] == 0: results.labels_[2] = 1
#         #        if results.labels_[3] == 0: results.labels_[3] = 1
#         #        if results.labels_[4] == 0: results.labels_[4] = 1
# 
#         # print(results.labels_[:50])
#         # count = 0
#         # for t in results.labels_[:50]:
#         #     if t == 0: count += 1
#         # print(count)
# 
#         print(feature)
# 
#         res_cent = np.array(results.cluster_centers_)
#         for cl in range(0, len(results.cluster_centers_)):
#             clu_cent[:, 0] += res_cent[:, 0]
#             clu_cent[:, 1] += res_cent[:, 1]
# 
#         clu_lab = results.labels_
#         PlotCluster(results.labels_, results.cluster_centers_, feature_sets[i])
#         i = i + 1
#         clusters = results.labels_.tolist()
#         score(clusters,  7)

