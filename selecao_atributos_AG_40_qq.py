# -*- coding: utf-8 -*-
import pandas as pd

print('hello world')
df = pd.read_csv('base_selecao_v2.csv',delimiter =',',encoding='utf-8') #importa a base



#separa a classe e exclui da base
y=df['Evasao']   
df= df.drop(columns=['Evasao'],axis=1) #separa a classe

print('base pronta')
tab_qui = pd.read_csv('chi_importancia.csv',delimiter =',',encoding='utf-8')
df=df[tab_qui.iloc[:40,0]]


#divisao da base 
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





#-----------  Função de custo

def getFitnessDT(individual, X, y):
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
        print("final conta")
        acc_val = (f1_score(y_testeGA, p,average='macro') - (num_feat/num_col)) 
        print('acc')
       # acc_val = accuracy_score(y_testeGA, p) - (num_feat/104) 
        ACC = (acc_val,)
        return  ACC 
     
def getFitnessRL(individual, X, y):
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
        acc_val = (f1_score(y_testeGA, p,average='macro') - (num_feat/num_col)) 
        print('acc')
       # acc_val = accuracy_score(y_testeGA, p) - (num_feat/104) 
        ACC = (acc_val,)
        return  ACC 
     
def getFitnessRF(individual, X, y):
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
        acc_val = (f1_score(y_testeGA, p,average='macro') - (num_feat/num_col)) 
        print('acc')
       # acc_val = accuracy_score(y_testeGA, p) - (num_feat/104) 
        ACC = (acc_val,)
        return  ACC 
     


     



#--------------------- Parâmeros

def geneticAlgorithmDT(X, y, n_population, n_generation):
    
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
    toolbox.register("evaluate",getFitnessDT, X=X, y=y)
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


def geneticAlgorithmRL(X, y, n_population, n_generation):
    
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
    toolbox.register("evaluate",getFitnessRL, X=X, y=y)
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


def geneticAlgorithmRF(X, y, n_population, n_generation):
    
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
    toolbox.register("evaluate",getFitnessRF, X=X, y=y)
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

hof = geneticAlgorithmDT(df, y, n_pop, n_gen)


# ------- Escolhe o vetor com melhor acurácia 

lista_acuracia = []
accuracy_dt, individual_dt,atrib_dt = bestIndividual(hof, df, y) 
print('estat pronta')
tab_selec= pd.DataFrame(atrib_dt)

fim2= time.time()
print("Atributos selecionados",atrib_dt)
print("Quantidade de atributos", len(atrib_dt))
print("Maior acuracia do genetico", accuracy_dt)

tab_selec.to_csv('atrib_AG_DT40_qq.csv',index=False)

time2.append(fim2-inicio2)
time5= pd.DataFrame(time2)
time5.to_csv('timeDT_AG40_qq.csv',index=False)

#---------

inicio2= time.time()

hof = geneticAlgorithmRL(df, y, n_pop, n_gen)


# ------- Escolhe o vetor com melhor acurácia 

lista_acuracia = []
accuracy_dt, individual_dt,atrib_dt = bestIndividual(hof, df, y) 
print('estat pronta')
tab_selec= pd.DataFrame(atrib_dt)
#X_train = X_train[atrib_dt]
#X_test = X_test[atrib_dt]
fim2= time.time()
print("Atributos selecionados",atrib_dt)
print("Quantidade de atributos", len(atrib_dt))
print("Maior acuracia do genetico", accuracy_dt)

tab_selec.to_csv('atrib_AG_RL40_qq.csv',index=False)

time2.append(fim2-inicio2)
time5= pd.DataFrame(time2)
time5.to_csv('timeRL_AG40_qq.csv',index=False)


#--------


inicio2= time.time()

hof = geneticAlgorithmRF(df, y, n_pop, n_gen)

# ------- Escolhe o vetor com melhor acurácia 

lista_acuracia = []
accuracy_dt, individual_dt,atrib_dt = bestIndividual(hof, df, y) 
print('estat pronta')
tab_selec= pd.DataFrame(atrib_dt)

fim2= time.time()
print("Atributos selecionados",atrib_dt)
print("Quantidade de atributos", len(atrib_dt))
print("Maior acuracia do genetico", accuracy_dt)

tab_selec.to_csv('atrib_AG_RF40_qq.csv',index=False)

time2.append(fim2-inicio2)
time5= pd.DataFrame(time2)
time5.to_csv('timeRF_AG40_qq.csv',index=False)


