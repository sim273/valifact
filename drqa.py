from sklearn.feature_extraction.text import TfidfVectorizer
import math
import nltk
nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import pandas as pd

tokenize = lambda doc: doc.lower().split(" ")
df1=pd.read_csv("dataset_original.csv")
#col_names =  ['CLAIM', 'EVIDENCE', 'LABEL', 'LABEL1']
df  = pd.DataFrame()
for index,row in df1.iterrows():
  print(row["SOURCE"],row["CLAIM"])
  file_name = 'stories/'
  file_name  += row['SOURCE'] + ".txt"
  print(file_name)
  with open(file_name, 'r+', encoding="utf-8") as f:
    s2 = f.read()
  sentence = row['CLAIM']
  df.insert(index,"CLAIM",row['CLAIM'])
  df.insert(index,"LABEL",row['LABEL'])
  df.insert(index,"LABEL1",row['LABEL1'])
  #print(sentence)
  s1=[sentence]
  s1+=nltk.sent_tokenize(s2)
  #print(s1)
  sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
  X = sklearn_tfidf.fit_transform(s1)
  #print(X)
  from sklearn.metrics.pairwise import cosine_similarity
  #print(X.shape)
  s=cosine_similarity(X[0:1], X)
  #print(s.shape)
    #res=np.asarray(np.where(s >= 0.05))
    #print(s)
  p = []
  q = []
  for i in s:
    for j in i:
      p.append(j)
      q.append(j)
    #print(p)
  index_i = []
  maxElements = []
  for i in range(4):
    max_ele=0
    index = 0
    index_max = index
    for j in p:
      if(j>max_ele):
        max_ele = j
        index_max = index
      index = index+1
    index_i.append(index_max)
    p[index_max]=0.0
    maxElements.append(max_ele)
      #print(index_i)
      #print(maxElements)
      #print(s1)
  s3=""
  for i in index_i:
      if(i!=0):
      #print(q[i])
       s3+=s1[i]
  df.insert(index,"EVIDENCE",s3)
df.to_csv("dataset_evidence.csv", encoding="utf-8")