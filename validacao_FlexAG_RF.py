# -*- coding: utf-8 -*-
import pandas as pd

df2 = pd.read_csv('base_validacao_v2.csv',delimiter =',',encoding='utf-8')

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
import time
import random
import numpy as np
from pandas.core import algorithms
from deap import creator, base, tools, algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import time

#separa a classe e exclui da base
y=df2['Evasao']   
df2= df2.drop(columns=['Evasao'],axis=1)


atrib_acad=list(np.array(pd.read_csv('atrib_AG_RF_acad_VC_.csv',delimiter =',',encoding='utf-8')).flatten())
atrib_inst=list(np.array(pd.read_csv('atrib_AG_RF_inst_VC_.csv',delimiter =',',encoding='utf-8')).flatten())
atrib_engaj=list(np.array(pd.read_csv('atrib_AG_RF_engaj_VC_.csv',delimiter =',',encoding='utf-8')).flatten())
atrib_financeiro=list(np.array(pd.read_csv('atrib_AG_RF_financeiro_VC_.csv',delimiter =',',encoding='utf-8')).flatten())
atrib_pessoal=list(np.array(pd.read_csv('atrib_AG_RF_pessoal_VC_.csv',delimiter =',',encoding='utf-8')).flatten())
atrib_curso=list(np.array(pd.read_csv('atrib_AG_RF_curso_VC_.csv',delimiter =',',encoding='utf-8')).flatten())

selecao=[atrib_acad,atrib_inst,atrib_engaj,atrib_financeiro,atrib_pessoal,atrib_curso]


Relatorio=[]

for i in range(0,6):
 df=df2[selecao[i]]

 Acuracia=[]  
 F1=[]
 Recall=[]
 Precisao=[]
 TempoClassi=[]
 TempoPred=[]



 skf=StratifiedKFold(n_splits=10,shuffle =True)
    #skf = KFold(n_splits=5)
 for train_index, test_index in skf.split(df,y):
     print("TRAIN:", train_index, "TEST:", test_index)
     X_treino, X_teste = df.iloc[train_index], df.iloc[test_index]
     y_treino, y_teste = y.iloc[train_index], y.iloc[test_index]
     inicio2= time.time()
     modelo = RandomForestClassifier(n_estimators=15)
     modelo =modelo.fit(X_treino, y_treino)
     fim2= time.time()
     inicio3= time.time()
     p= modelo.predict(X_teste)
     fim3= time.time()
     acc = accuracy_score(y_teste, p)
     f1=f1_score(y_teste, p,average='macro')
     precisao= precision_score(y_teste, p,average='macro')
     recall=recall_score(y_teste, p,average='macro')
     Acuracia.append(acc)  
     F1.append(f1)
     Recall.append(recall)
     Precisao.append(precisao)
     TempoClassi.append(fim2-inicio2)
     TempoPred.append(fim3-inicio3)




 print("Acc",Acuracia,np.array(Acuracia).mean())
 print("F1 ",F1,np.array(F1).mean())
 print("Precisao",Precisao,np.array(Precisao).mean())
 print("Recall ",Recall,np.array(Recall).mean())
 print("tempoClassi",TempoClassi,np.array(TempoClassi).mean())
 print("tempoPred",TempoPred,np.array(TempoPred).mean()) 



 Relatorio.append(Acuracia)
 Relatorio.append(F1)
 Relatorio.append(Recall)
 Relatorio.append(Precisao)
 Relatorio.append(TempoClassi)
 Relatorio.append(TempoPred)
 Relatorio.append(i)

print(Relatorio)
np.savetxt('Classi_peso_RF_peso.csv',Relatorio, delimiter=',',fmt='%s')