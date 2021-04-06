# source : https://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/
# https://scikit-learn.org/stable/modules/clustering.html
# https://code.google.com/archive/p/word2vec/

from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn import cluster
from collections import defaultdict

with open("article_test.txt") as f:
    text = f.read()

stopwords = stopwords.words("english")


model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

# EXCLURE NAMED ENTITIES TODO

tok_text = [
    tok
    for tok in set(word_tokenize(text))
    if tok in model and tok.lower() not in stopwords
]

voc_matrix = np.array([model[tok] for tok in tok_text])


## K-means clustering
clustering = cluster.AffinityPropagation(random_state=5).fit(voc_matrix)

clusters = defaultdict(list)

for word, label in zip(tok_text, clustering.labels_):
    clusters[label].append(word)

## Extended lexical field (most_similar)
for i, words in sorted(clusters.items()):
    if len(words) > 2:
        print(f"Champ lexical {i} : {words}")
        print(
            f"Mots proches de ce champ lexical : {[word for word, _ in model.most_similar(positive=words, topn=min(10,len(words)))]}"  # TODO proposer des mots + courants
        )
        print()

# TODO GRIDSEARCH POUR OPTIMISER CHOIX CLUSTERING & PARAMETRES