# text-analysis-lex-fridman
ML - Podcast Transcript Analysis with Python - Lex Fridman 

## Lex Fridman Podcast Transcript Analysis with Python

## Importing libraries
``
import numpy as np 
import pandas as pd 
``
## Input data files are available in the read-only "../input/" directory

``
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
`` 
## This dataset features discussions with thought leaders from diverse fields such as technology, science, philosophy, and art, this dataset offers a treasure trove of insights and wisdom. Exploring the nuances of each conversation can uncover emerging trends. Exploring some concepts below: 

``
data=pd.read_csv('/kaggle/input/lex-fridman-podcast-transcript/podcastdata_dataset.csv')
``

``
data.head()
``

<img width="581" alt="Screenshot 2024-04-19 at 1 15 12 AM" src="https://github.com/saheelchowdhury/text-analysis-lex-fridman/assets/153671296/efc4f727-d481-4edb-b6f3-83f0cc196dd1">

``
from gensim.models.word2vec import Word2Vec
``


# Data Cleaning

``
import re
from sklearn import feature_extraction 
stop_words = feature_extraction.text.ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
def preprocess(text):
  text = text.lower() #lowercase
  text = re.sub(r'[^\w\s]', '', text) 
  text = re.sub(r'\d+', '', text) 
  text = " ".join(text.split()) 
  text = text.split()
  text = [x for x in text if x not in stop_words] 
  text = [x for x in text if x not in ["dr", "doctor"]] 
  text = " ".join(text)
  return(text)
``

## Picking the text column
``
data['review_processed']=data['text'].apply(lambda x:preprocess(x))
data['review_processed']=data['review_processed'].apply(lambda x:x.split())
model = Word2Vec(sentences=data['review_processed'].tolist(), vector_size=100, sg=1,min_count=5,window=5,workers=50,seed=10,epochs=50)
``
## Saving model
``
model.save('w2v_dr.w2v')
``
``
model=Word2Vec.load('w2v_dr.w2v')
``
``
vocab = model.wv.index_to_key
``

## Testing - Similar Words to 'drugs'
``
model.wv.most_similar('drugs', topn=10)
``

<img width="392" alt="Screenshot 2024-04-19 at 1 21 43 AM" src="https://github.com/saheelchowdhury/text-analysis-lex-fridman/assets/153671296/5a9210ab-75b7-4f06-92d2-64753495ab39">

## Testing - Similar Words to 'race'
``
model.wv.most_similar('race', topn=10)
``

## Exploring the relationship regarding the 'interesting' exclamation with regards to topic names

``
print(model.wv.similarity('rich', 'interesting'))
print(model.wv.similarity('artificial', 'interesting'))
print(model.wv.similarity('psychology', 'interesting'))
print(model.wv.similarity('race', 'interesting'))
``

<img width="514" alt="Screenshot 2024-04-19 at 1 23 37 AM" src="https://github.com/saheelchowdhury/text-analysis-lex-fridman/assets/153671296/5e67ce15-e6bd-4184-9cc5-cf810c98f25e">

## Finding similar sentiments/topics related to capitalism - related discussions on the podcast

``
print(model.wv.similarity('mistake', 'capitalism'))
print(model.wv.similarity('race', 'capitalism'))
print(model.wv.similarity('global', 'capitalism'))
print(model.wv.similarity('war', 'capitalism'))
``
<img width="563" alt="Screenshot 2024-04-19 at 1 24 18 AM" src="https://github.com/saheelchowdhury/text-analysis-lex-fridman/assets/153671296/dc4abe53-5e69-46c8-92eb-b512070d7270">

## War is a possible topic of discussion on the show when the words 'race' & 'religion' are used (0.32 cosine similarity) 

``
v_war=model.wv['war']
v_race = model.wv['race']
v_religion = model.wv['religion']
created_st = v_race  + v_religion
np.dot(created_st, v_war)/(np.linalg.norm(created_st)* np.linalg.norm(v_war))
``
## 0.3267056

## Nuclear war is a possible 'issue' discussed while talking about 'Russia' (high cosine similarity of .442) 

``
v_russia = model.wv['russia']
v_issues = model.wv['issues']
v_nuclear = model.wv['nuclear']
created_nuclear = v_russia + v_issues
np.dot(created_nuclear, v_nuclear)/(np.linalg.norm(created_nuclear)* np.linalg.norm(v_nuclear))
``
## 0.44290972




