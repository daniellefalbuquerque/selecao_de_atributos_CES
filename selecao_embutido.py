# -*- coding: utf-8 -*-
import pandas as pd
import time
df = pd.read_csv('base_selecao_v2.csv',delimiter =',',encoding='utf-8')

print("TAMANHO",df.shape)


#separa a classe e exclui da base
y=df['Evasao']   
df= df.drop(columns=['Evasao'],axis=1)

#print(y.value_counts())



from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier

inicio = time.time()
modelo = DecisionTreeClassifier()
modelo =modelo.fit(df, y)
importancia=modelo.feature_importances_  #importancia da DT
tab_importancia = []
tab_importancia.append(df.columns)
tab_importancia.append(importancia)
tab_importancia= pd.DataFrame(tab_importancia)
tab_importancia =tab_importancia.transpose()
tab_importancia = tab_importancia.sort_values(1,ascending=False)
fim = time.time()

print("DT",tab_importancia)
print('tempoDT', fim-inicio)

#grafico
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
tab_importancia.to_csv('dt_importancia.csv',index=False)



import pandas as pd
import time
df = pd.read_csv('base_selecao_v2.csv',delimiter =',',encoding='utf-8')

print("TAMANHO",df.shape)


#separa a classe e exclui da base
y=df['Evasao']   
df= df.drop(columns=['Evasao'],axis=1)

#print(y.value_counts())



from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


inicio = time.time()

lr= LogisticRegression(max_iter=200, random_state=42,solver='sag',C=0.5)

modelo = lr.fit(df, y)


importancia=abs(modelo.coef_[0]) #importancia da RL
tab_importancia = []
tab_importancia.append(df.columns)
tab_importancia.append(importancia)
tab_importancia= pd.DataFrame(tab_importancia)
tab_importancia =tab_importancia.transpose()
tab_importancia = tab_importancia.sort_values(1,ascending=False)
fim = time.time()

print("RL",tab_importancia)
tab_importancia.to_csv('rl_importancia.csv',index=False)
print('tempoRL', fim-inicio)

import pandas as pd
import time
df = pd.read_csv('base_selecao_v2.csv',delimiter =',',encoding='utf-8')

print("TAMANHO",df.shape)


#separa a classe e exclui da base
y=df['Evasao']   
df= df.drop(columns=['Evasao'],axis=1)

 


inicio = time.time()
modelo = RandomForestClassifier(n_estimators=15)
modelo =modelo.fit(df, y)
importancia=modelo.feature_importances_  #importancia da DT
tab_importancia = []
tab_importancia.append(df.columns)
tab_importancia.append(importancia)
tab_importancia= pd.DataFrame(tab_importancia)
tab_importancia =tab_importancia.transpose()
tab_importancia = tab_importancia.sort_values(1,ascending=False)
fim = time.time()
print('tempoRF', fim-inicio)



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
tab_importancia.to_csv('rf_importancia.csv',index=False)
