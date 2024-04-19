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
