# -*- coding: utf-8 -*-
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv('base_selecao_v2.csv',delimiter =',',encoding='utf-8')

print("TAMANHO",df.shape)

#separa a classe e exclui da base
y=df['Evasao']  

print(y.value_counts())


inicio = time.time()
corre=df.corr()
corre=corre['Evasao']
corre=abs(corre) #modulo da correlacao

tab_corre = []
tab_corre.append(df.columns)
tab_corre.append(corre)
tab_corre= pd.DataFrame(tab_corre)
tab_corre =tab_corre.transpose() 
tab_corre = tab_corre.sort_values(1,ascending=False) #ordena
tab_corre = tab_corre.iloc[1:]

fim = time.time()

print("tempo",fim-inicio)
print("Correlacao",tab_corre)
tab_corre.to_csv('corre_importancia.csv',index=False)
