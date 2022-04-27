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

df= df.drop(columns=['Evasao'],axis=1)


#----- Qui
from sklearn.feature_selection import SelectKBest, chi2

#ranking dos atributos qui quadrado

inicio = time.time()
quadro_chi= chi2(df,y)
qui = quadro_chi[0]
tab_qui = []
tab_qui.append(df.columns)
tab_qui.append(qui)
tab_qui= pd.DataFrame(tab_qui)
tab_qui =tab_qui.transpose()
tab_qui = tab_qui.sort_values(1,ascending=False)

fim = time.time()

print("Qui-quiadrado",tab_qui)
print('tempo', fim-inicio)
tab_qui.to_csv('chi_importancia.csv',index=False)


