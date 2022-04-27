
# -*- coding: utf-8 -*-
import pandas as pd

print('hello world')
df = pd.read_csv('base_selecao_v2.csv',delimiter =',',encoding='utf-8') #importa a base


#separa a classe e exclui da base
y=df['Evasao']   
df= df.drop(columns=['Evasao'],axis=1) #separa a classe

print('base pronta')
#tab_corre = pd.read_csv('corre_importancia.csv',delimiter =',',encoding='utf-8')

from sklearn.model_selection import train_test_split #divide em treino e teste pro genetico



print('base dividida')
#importa bib
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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier



n_pop = 15 #parametros genetico
n_gen = 15


#DT


#-------------cria pesos--------------------#

def cria_vetor_pesos_academico(categ,df_colunas):
  for i in range(len(data)):
    for j in range(len(categ)):
      if data[i]== categ.iloc[j,0]:
       if categ.iloc[j,1] =="Institucional":
         vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Engajamento":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Academico":
          vetorpeso[i]= 0.95
       if categ.iloc[j,1] =="Pessoal":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Financeiro":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Curso":
          vetorpeso[i]= 0.01

def cria_vetor_pesos_institucional(categ,df_colunas):
  for i in range(len(data)):
    for j in range(len(categ)):
      if data[i]== categ.iloc[j,0]:
       if categ.iloc[j,1] =="Institucional":
         vetorpeso[i]= 0.95
       if categ.iloc[j,1] =="Engajamento":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Academico":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Pessoal":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Financeiro":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Curso":
          vetorpeso[i]= 0.01

def cria_vetor_pesos_engajamento(categ,df_colunas):
  for i in range(len(data)):
    for j in range(len(categ)):
      if data[i]== categ.iloc[j,0]:
       if categ.iloc[j,1] =="Institucional":
         vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Engajamento":
          vetorpeso[i]= 0.95
       if categ.iloc[j,1] =="Academico":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Pessoal":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Financeiro":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Curso":
          vetorpeso[i]= 0.01

def cria_vetor_pesos_financeiro(categ,df_colunas):
  for i in range(len(data)):
    for j in range(len(categ)):
      if data[i]== categ.iloc[j,0]:
       if categ.iloc[j,1] =="Institucional":
         vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Engajamento":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Academico":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Pessoal":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Financeiro":
          vetorpeso[i]= 0.95
       if categ.iloc[j,1] =="Curso":
          vetorpeso[i]= 0.01

def cria_vetor_pesos_pessoal(categ,df_colunas):
  for i in range(len(data)):
    for j in range(len(categ)):
      if data[i]== categ.iloc[j,0]:
       if categ.iloc[j,1] =="Institucional":
         vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Engajamento":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Academico":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Pessoal":
          vetorpeso[i]= 0.95
       if categ.iloc[j,1] =="Financeiro":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Curso":
          vetorpeso[i]= 0.01

def cria_vetor_pesos_curso(categ,df_colunas):
  for i in range(len(data)):
    for j in range(len(categ)):
      if data[i]== categ.iloc[j,0]:
       if categ.iloc[j,1] =="Institucional":
         vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Engajamento":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Academico":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Pessoal":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Financeiro":
          vetorpeso[i]= 0.01
       if categ.iloc[j,1] =="Curso":
          vetorpeso[i]= 0.95
#-----------  Função de custo

def getFitnessDF(individual, X, y,vetorpeso):
    prob=vetorpeso
    num_col=len(X.columns)
    vetorbase = [1]*num_col

    if (individual.count(0) != len(individual)):   #varre o vetor 
        cols = [index for index in range(
            len(individual)) if individual[index] == 0] #vetor com os indices das colunas onde o valor é zero em individual

        # get features subset
        X_subset = X.drop(X.columns[cols], axis=1)  #exclui da base as colunas de indice cols
        X_treinoGA, X_testeGA, y_treinoGA, y_testeGA = train_test_split(X_subset,y,test_size=0.2)
        num_feat = len(X_subset.columns)
        print('antes classi gen')
        # apply classification algorithm
        modelo =DecisionTreeClassifier()
        modelo.fit(X_treinoGA, y_treinoGA)
        p= modelo.predict(X_testeGA)
        print('after')
        col = np.zeros(num_col)
        for i in range(len(col)):
          if i in cols:
            col[i]=0
          else:
            col[i]=1
        print("col")
        prod= col @ prob
        deno= np.array(vetorbase) @ prob 
        valor=prod/deno
        print("final conta")
        acc_val = (f1_score(y_testeGA, p,average='macro') - (num_feat/num_col)+valor) 
        print('acc')
        ACC = (acc_val,)
        return  ACC 
     
def getFitnessRL(individual, X, y,vetorpeso):
    prob=vetorpeso
    num_col=len(X.columns)
    vetorbase = [1]*num_col

    if (individual.count(0) != len(individual)):   #varre o vetor 
        cols = [index for index in range(
            len(individual)) if individual[index] == 0] #vetor com os indices das colunas onde o valor é zero em individual

        # get features subset
        X_subset = X.drop(X.columns[cols], axis=1)  #exclui da base as colunas de indice cols
        X_treinoGA, X_testeGA, y_treinoGA, y_testeGA = train_test_split(X_subset,y,test_size=0.2)
        num_feat = len(X_subset.columns)
        print('antes classi gen')
        # apply classification algorithm
        modelo =LogisticRegression(max_iter=200, random_state=42,solver='sag',C=0.5)
        modelo.fit(X_treinoGA, y_treinoGA)
        p= modelo.predict(X_testeGA)
        print('after')
        col = np.zeros(num_col)
        for i in range(len(col)):
          if i in cols:
            col[i]=0
          else:
            col[i]=1
        print("col")
        prod= col @ prob
        deno= np.array(vetorbase) @ prob 
        valor=prod/deno
        print("final conta")
        acc_val = (f1_score(y_testeGA, p,average='macro') - (num_feat/num_col)+valor) 
        print('acc') 
        ACC = (acc_val,)
        return  ACC 
     
def getFitness(individual, X, y,vetorpeso):
    prob=vetorpeso
    num_col=len(X.columns)
    vetorbase = [1]*num_col

    if (individual.count(0) != len(individual)):   #varre o vetor 
        cols = [index for index in range(
            len(individual)) if individual[index] == 0] #vetor com os indices das colunas onde o valor é zero em individual

        # get features subset
        X_subset = X.drop(X.columns[cols], axis=1)  #exclui da base as colunas de indice cols
        X_treinoGA, X_testeGA, y_treinoGA, y_testeGA = train_test_split(X_subset,y,test_size=0.2)
        num_feat = len(X_subset.columns)
        print('antes classi gen')
        # apply classification algorithm
        modelo =RandomForestClassifier(n_estimators=15)
        modelo.fit(X_treinoGA, y_treinoGA)
        p= modelo.predict(X_testeGA)
        print('after')
        col = np.zeros(num_col)
        for i in range(len(col)):
          if i in cols:
            col[i]=0
          else:
            col[i]=1
        print("col")
        prod= col @ prob
        deno= np.array(vetorbase) @ prob 
        valor=prod/deno
        print("final conta")
        acc_val = (f1_score(y_testeGA, p,average='macro') - (num_feat/num_col)+valor) 
        print('acc') 
        ACC = (acc_val,)
        return  ACC 
     





#--------------------- Parâmeros

def geneticAlgorithm(X, y, n_population, n_generation,vetorpeso):
    
    # Cria individuo
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) 
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Cria ferramentas
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate",getFitness, X=X, y=y,vetorpeso=vetorpeso)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    #imputa parâmetros
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Algoritmo genético
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.01,
                                   ngen=n_generation, stats=stats, halloffame=hof, verbose=True)
    
    return hof



def bestIndividual(hof, X, y):
    maxAccurcy = (0,)
    for individual in hof:
        lista_acuracia.append(individual.fitness.values)      
        _individual = hof[0]  
        individualAtrib = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]  #separa apenas as colunas com 1
    return _individual.fitness.values, _individual, individualAtrib # descobrir o _individual_header



#-------inicio

time2=[]
#------------------DT



inicio2= time.time()

categ = pd.read_csv('categv2.csv',delimiter =',',encoding='utf-8')

vetorpeso = [0]*len(df.columns)
df_colunas=df.columns
data = np.array(df_colunas)
cria_vetor_pesos_academico(categ,df)
print('vetor data',data)
print('academico')
print('vetor peso',vetorpeso)
hof = geneticAlgorithm(df, y, n_pop, n_gen,vetorpeso=vetorpeso)
print('gen pronto')
print("HAL DA FAMA",hof)
print('resultado',hof)

# ------- Escolhe o vetor com melhor acurácia 

lista_acuracia = []
accuracy_dt, individual_dt,atrib_dt = bestIndividual(hof,df, y) 
print('estat pronta')
tab_selec= pd.DataFrame(atrib_dt)

fim2= time.time()
print("Atributos selecionados",atrib_dt)
print("Quantidade de atributos", len(atrib_dt))
print("Maior acuracia do genetico", accuracy_dt)
tab_selec.to_csv('atrib_AG_RF_acad_VC_.csv',index=False)
time2.append(fim2-inicio2)
time5= pd.DataFrame(time2)
time5.to_csv('timeRF.csv',index=False)



inicio2= time.time()

categ = pd.read_csv('categv2.csv',delimiter =',',encoding='utf-8')

vetorpeso = [0]*len(df.columns)
df_colunas=df.columns
data = np.array(df_colunas)
cria_vetor_pesos_institucional(categ,df)
hof = geneticAlgorithm(df, y, n_pop, n_gen,vetorpeso=vetorpeso)
print('gen pronto')
print("HAL DA FAMA",hof)
print('resultado',hof)

# ------- Escolhe o vetor com melhor acurácia 

lista_acuracia = []
accuracy_dt, individual_dt,atrib_dt = bestIndividual(hof, df, y) 
print('estat pronta')
tab_selec= pd.DataFrame(atrib_dt)

fim2= time.time()
print("Atributos selecionados",atrib_dt)
print("Quantidade de atributos", len(atrib_dt))
print("Maior acuracia do genetico", accuracy_dt)

tab_selec.to_csv('atrib_AG_RF_inst_VC_.csv',index=False)

time2.append(fim2-inicio2)
time5= pd.DataFrame(time2)
time5.to_csv('timeRF.csv',index=False)


inicio2= time.time()

categ = pd.read_csv('categv2.csv',delimiter =',',encoding='utf-8')

vetorpeso = [0]*len(df.columns)
df_colunas=df.columns
data = np.array(df_colunas)
cria_vetor_pesos_engajamento(categ,df)
hof = geneticAlgorithm(df, y, n_pop, n_gen,vetorpeso=vetorpeso)
print('gen pronto')
print("HAL DA FAMA",hof)
print('resultado',hof)

# ------- Escolhe o vetor com melhor acurácia 

lista_acuracia = []
accuracy_dt, individual_dt,atrib_dt = bestIndividual(hof, df, y) 
print('estat pronta')
tab_selec= pd.DataFrame(atrib_dt)

fim2= time.time()
print("Atributos selecionados",atrib_dt)
print("Quantidade de atributos", len(atrib_dt))
print("Maior acuracia do genetico", accuracy_dt)

tab_selec.to_csv('atrib_AG_RF_engaj_VC_.csv',index=False)

time2.append(fim2-inicio2)
time5= pd.DataFrame(time2)
time5.to_csv('timeRF.csv',index=False)


inicio2= time.time()

categ = pd.read_csv('categv2.csv',delimiter =',',encoding='utf-8')

vetorpeso = [0]*len(df.columns)
df_colunas=df.columns
data = np.array(df_colunas)
cria_vetor_pesos_financeiro(categ,df)
hof = geneticAlgorithm(df, y, n_pop, n_gen,vetorpeso=vetorpeso)
print('gen pronto')
print("HAL DA FAMA",hof)
print('resultado',hof)

# ------- Escolhe o vetor com melhor acurácia 

lista_acuracia = []
accuracy_dt, individual_dt,atrib_dt = bestIndividual(hof, df, y) 
print('estat pronta')
tab_selec= pd.DataFrame(atrib_dt)

fim2= time.time()
print("Atributos selecionados",atrib_dt)
print("Quantidade de atributos", len(atrib_dt))
print("Maior acuracia do genetico", accuracy_dt)

tab_selec.to_csv('atrib_AG_RF_financeiro_VC_.csv',index=False)

time2.append(fim2-inicio2)
time5= pd.DataFrame(time2)
time5.to_csv('timeRF.csv',index=False)


inicio2= time.time()

categ = pd.read_csv('categv2.csv',delimiter =',',encoding='utf-8')

vetorpeso = [0]*len(df.columns)
df_colunas=df.columns
data = np.array(df_colunas)
cria_vetor_pesos_pessoal(categ,df)
hof = geneticAlgorithm(df, y, n_pop, n_gen,vetorpeso=vetorpeso)
print('gen pronto')
print("HAL DA FAMA",hof)
print('resultado',hof)

# ------- Escolhe o vetor com melhor acurácia 

lista_acuracia = []
accuracy_dt, individual_dt,atrib_dt = bestIndividual(hof, df, y) 
print('estat pronta')
tab_selec= pd.DataFrame(atrib_dt)

fim2= time.time()
print("Atributos selecionados",atrib_dt)
print("Quantidade de atributos", len(atrib_dt))
print("Maior acuracia do genetico", accuracy_dt)

tab_selec.to_csv('atrib_AG_RF_pessoal_VC_.csv',index=False)

time2.append(fim2-inicio2)
time5= pd.DataFrame(time2)
time5.to_csv('timeRF.csv',index=False)


inicio2= time.time()

categ = pd.read_csv('categv2.csv',delimiter =',',encoding='utf-8')

vetorpeso = [0]*len(df.columns)
df_colunas=df.columns
data = np.array(df_colunas)
cria_vetor_pesos_curso(categ,df)
hof = geneticAlgorithm(df, y, n_pop, n_gen,vetorpeso=vetorpeso)
print('gen pronto')
print("HAL DA FAMA",hof)
print('resultado',hof)

# ------- Escolhe o vetor com melhor acurácia 

lista_acuracia = []
accuracy_dt, individual_dt,atrib_dt = bestIndividual(hof, df, y) 
print('estat pronta')
tab_selec= pd.DataFrame(atrib_dt)

fim2= time.time()
print("Atributos selecionados",atrib_dt)
print("Quantidade de atributos", len(atrib_dt))
print("Maior acuracia do genetico", accuracy_dt)

tab_selec.to_csv('atrib_AG_RF_curso_VC_.csv',index=False)

time2.append(fim2-inicio2)
time5= pd.DataFrame(time2)
time5.to_csv('timeRF.csv',index=False)