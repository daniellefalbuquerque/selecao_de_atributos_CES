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
from sklearn.linear_model import LogisticRegression
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




atrib_DT=np.array(pd.read_csv('dt_importancia.csv',delimiter =',',encoding='utf-8',usecols=[0]))
atrib_RL=np.array(pd.read_csv('rl_importancia.csv',delimiter =',',encoding='utf-8',usecols=[0]))
atrib_RF=np.array(pd.read_csv('rf_importancia.csv',delimiter =',',encoding='utf-8',usecols=[0]))


atrib_DT=list(atrib_DT.flatten())
atrib_RL=list(atrib_RL.flatten())
atrib_RF=list(atrib_RF.flatten())


print(atrib_RL)
print(atrib_DT)
print(atrib_RF)



Relatorio=[]

print('atrib')
atrib_DT=atrib_DT[:20]
df=df2[atrib_DT]

Acuracia=[]  
F1=[]
Recall=[]
Precisao=[]
TempoClassi=[]
TempoPred=[]

print("amostra dados")
print(df)

skf=StratifiedKFold(n_splits=10,shuffle =True)
for train_index, test_index in skf.split(df,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_treino, X_teste = df.iloc[train_index], df.iloc[test_index]
    y_treino, y_teste = y.iloc[train_index], y.iloc[test_index]
    inicio2= time.time()
    modelo = DecisionTreeClassifier()
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
#Relatorio.append(i)

print(Relatorio)

np.savetxt('Embutido20.csv',Relatorio, delimiter=',',fmt='%s')

atrib_RL=atrib_RL[:20]
df=df2[atrib_RL]


Acuracia=[]  
F1=[]
Recall=[]
Precisao=[]
TempoClassi=[]
TempoPred=[]

print("amostra dados")
print(df)


skf=StratifiedKFold(n_splits=10,shuffle =True)
    #skf = KFold(n_splits=5)
for train_index, test_index in skf.split(df,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_treino, X_teste = df.iloc[train_index], df.iloc[test_index]
    y_treino, y_teste = y.iloc[train_index], y.iloc[test_index]
    inicio2= time.time()
    modelo = LogisticRegression(max_iter=200, random_state=42,solver='sag',C=0.5)
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
#Relatorio.append(i)

print(Relatorio)

np.savetxt('Embutido20.csv',Relatorio, delimiter=',',fmt='%s')


#------------

atrib_RF=atrib_RF[:20]
df=df2[atrib_RF]


Acuracia=[]  
F1=[]
Recall=[]
Precisao=[]
TempoClassi=[]
TempoPred=[]

print("amostra dados")
print(df)


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
#Relatorio.append(i)

print(Relatorio)
#Relatorio= pd.DataFrame(Relatorio)
#Relatorio.to_csv('Classi_peso_DT40.csv',index=False,sep=",")
np.savetxt('Embutido20.csv',Relatorio, delimiter=',',fmt='%s')


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
from sklearn.linear_model import LogisticRegression
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




atrib_DT=np.array(pd.read_csv('dt_importancia.csv',delimiter =',',encoding='utf-8',usecols=[0]))
atrib_RL=np.array(pd.read_csv('rl_importancia.csv',delimiter =',',encoding='utf-8',usecols=[0]))
atrib_RF=np.array(pd.read_csv('rf_importancia.csv',delimiter =',',encoding='utf-8',usecols=[0]))


atrib_DT=list(atrib_DT.flatten())
atrib_RL=list(atrib_RL.flatten())
atrib_RF=list(atrib_RF.flatten())



Relatorio=[]

print('atrib')
atrib_DT=atrib_DT[:40]
df=df2[atrib_DT]

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
    modelo = DecisionTreeClassifier()
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
#Relatorio.append(i)

print(Relatorio)

np.savetxt('Embutido40.csv',Relatorio, delimiter=',',fmt='%s')

atrib_RL=atrib_RL[:40]
df=df2[atrib_RL]


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
    modelo = LogisticRegression(max_iter=200, random_state=42,solver='sag',C=0.5)
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
#Relatorio.append(i)

print(Relatorio)

np.savetxt('Embutido40.csv',Relatorio, delimiter=',',fmt='%s')


#------------

atrib_RF=atrib_RF[:40]
df=df2[atrib_RF]


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
#Relatorio.append(i)

print(Relatorio)
#Relatorio= pd.DataFrame(Relatorio)
#Relatorio.to_csv('Classi_peso_DT40.csv',index=False,sep=",")
np.savetxt('Embutido40.csv',Relatorio, delimiter=',',fmt='%s')




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
from sklearn.linear_model import LogisticRegression
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



atrib_DT=np.array(pd.read_csv('dt_importancia.csv',delimiter =',',encoding='utf-8',usecols=[0]))
atrib_RL=np.array(pd.read_csv('rl_importancia.csv',delimiter =',',encoding='utf-8',usecols=[0]))
atrib_RF=np.array(pd.read_csv('rf_importancia.csv',delimiter =',',encoding='utf-8',usecols=[0]))


atrib_DT=list(atrib_DT.flatten())
atrib_RL=list(atrib_RL.flatten())
atrib_RF=list(atrib_RF.flatten())



Relatorio=[]

print('atrib')
atrib_DT=atrib_DT[:60]
df=df2[atrib_DT]
df.head(5)

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
    modelo = DecisionTreeClassifier()
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
#Relatorio.append(i)

print(Relatorio)
#Relatorio= pd.DataFrame(Relatorio)
#Relatorio.to_csv('Classi_peso_DT40.csv',index=False,sep=",")
np.savetxt('Embutido60.csv',Relatorio, delimiter=',',fmt='%s')

atrib_RL=atrib_RL[:60]
df=df2[atrib_RL]


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
    modelo = LogisticRegression(max_iter=200, random_state=42,solver='sag',C=0.5)
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
#Relatorio.append(i)

print(Relatorio)
#Relatorio= pd.DataFrame(Relatorio)
#Relatorio.to_csv('Classi_peso_DT40.csv',index=False,sep=",")
np.savetxt('Embutido60.csv',Relatorio, delimiter=',',fmt='%s')


#------------

atrib_RF=atrib_RF[:60]
df=df2[atrib_RF]


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
#Relatorio.append(i)

print(Relatorio)

np.savetxt('Embutido60.csv',Relatorio, delimiter=',',fmt='%s')



