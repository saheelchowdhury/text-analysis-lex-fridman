# text-analysis-lex-fridman
ML - Podcast Transcript Analysis with Python - Lex Fridman 

## Lex Fridman Podcast Transcript

## Importing libraries
``
import numpy as np 
import pandas as pd 
``
# Input data files are available in the read-only "../input/" directory

``
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
`` 
