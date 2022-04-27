import pandas as pd

# base de latitude e longitude dos municipio
locali = pd.read_csv('locali.csv',delimiter =';',encoding='latin-1',usecols=['latitude','longitude','CO_MUNICIPIO','codigo_uf'])



#-------------------------------IES


df_ies = pd.read_csv('DM_IES.CSV',delimiter ='|',encoding='latin-1',usecols=['CO_IES','NO_IES','QT_TEC_TOTAL','IN_ACESSO_PORTAL_CAPES','IN_ACESSO_OUTRAS_BASES','IN_REPOSITORIO_INSTITUCIONAL','IN_BUSCA_INTEGRADA','IN_SERVICO_INTERNET','IN_PARTICIPA_REDE_SOCIAL','IN_CATALOGO_ONLINE','QT_PERIODICO_ELETRONICO','QT_LIVRO_ELETRONICO','TP_REFERENTE','VL_RECEITA_PROPRIA','VL_RECEITA_TRANSFERENCIA','VL_RECEITA_OUTRA','VL_DESPESA_PESSOAL_DOCENTE','VL_DESPESA_PESSOAL_TECNICO','VL_DESPESA_PESSOAL_ENCARGO','VL_DESPESA_CUSTEIO','VL_DESPESA_INVESTIMENTO','VL_DESPESA_PESQUISA','VL_DESPESA_OUTRA'])


#novos atributos em funcao
df_ies['receitas']= (df_ies['VL_RECEITA_PROPRIA']+df_ies['VL_RECEITA_TRANSFERENCIA']+df_ies['VL_RECEITA_OUTRA'])

df_ies['despesas']= (df_ies['VL_DESPESA_PESSOAL_DOCENTE']+df_ies['VL_DESPESA_PESSOAL_TECNICO']+df_ies['VL_DESPESA_PESSOAL_ENCARGO']+df_ies['VL_DESPESA_CUSTEIO']+df_ies['VL_DESPESA_INVESTIMENTO']+df_ies['VL_DESPESA_PESQUISA'] + df_ies['VL_DESPESA_OUTRA'])


df_ies['per_pesquisa']=(df_ies['VL_DESPESA_PESQUISA'] / df_ies['despesas'])
df_ies['per_investimento']=(df_ies['VL_DESPESA_INVESTIMENTO'] / df_ies['despesas'])


df_ies.drop_duplicates()



#-------------curso

df_curso = pd.read_csv('DM_CURSO.CSV',delimiter =',',encoding='latin-1',usecols=['CO_CURSO','CO_UF','CO_MUNICIPIO','IN_CAPITAL','NO_CURSO','IN_GRATUITO','TP_ATRIBUTO_INGRESSO','NU_CARGA_HORARIA','IN_AJUDA_DEFICIENTE','IN_OFERECE_DISC_SEMI_PRES','QT_MATRICULA_TOTAL','QT_CONCLUINTE_TOTAL','QT_INGRESSO_TOTAL','QT_INGRESSO_VAGA_NOVA','QT_VAGA_TOTAL','CO_LOCAL_OFERTA'])



#dados ausentes dos cursos de ead
df_curso.update(df_curso['CO_UF'].fillna(00))
df_curso.update(df_curso['CO_MUNICIPIO'].fillna(0))
df_curso.update(df_curso['IN_CAPITAL'].fillna(00))


#nao faz sentido para ead
df_curso=df_curso.drop(columns=['IN_OFERECE_DISC_SEMI_PRES','CO_LOCAL_OFERTA'],axis=1)


#exclui os cursos que nao preencheram esses atributos
df_curso= df_curso.dropna(subset=['QT_MATRICULA_TOTAL','QT_CONCLUINTE_TOTAL','QT_INGRESSO_TOTAL','QT_INGRESSO_VAGA_NOVA','QT_VAGA_TOTAL','IN_AJUDA_DEFICIENTE','IN_GRATUITO'])

df_curso.drop_duplicates()





#---------Aluno

df = pd.read_csv('DM_ALUNO.CSV',delimiter ='|',low_memory=False,usecols=['NU_ANO_CENSO','CO_IES','TP_CATEGORIA_ADMINISTRATIVA','TP_ORGANIZACAO_ACADEMICA','CO_CURSO','CO_CURSO_POLO','TP_TURNO','TP_GRAU_ACADEMICO','TP_MODALIDADE_ENSINO','TP_NIVEL_ACADEMICO','CO_OCDE','TP_COR_RACA','TP_SEXO','NU_IDADE','TP_NACIONALIDADE','CO_PAIS_ORIGEM','CO_UF_NASCIMENTO','CO_MUNICIPIO_NASCIMENTO','TP_DEFICIENCIA','TP_SITUACAO','QT_CARGA_HORARIA_TOTAL','DT_INGRESSO_CURSO','IN_INGRESSO_VESTIBULAR','IN_INGRESSO_ENEM','IN_INGRESSO_AVALIACAO_SERIADA','IN_INGRESSO_SELECAO_SIMPLIFICA','IN_INGRESSO_OUTRO_TIPO_SELECAO','IN_INGRESSO_VAGA_REMANESC','IN_INGRESSO_VAGA_PROG_ESPECIAL','IN_INGRESSO_TRANSF_EXOFFICIO','IN_INGRESSO_DECISAO_JUDICIAL','IN_INGRESSO_CONVENIO_PECG','IN_INGRESSO_EGRESSO','IN_INGRESSO_OUTRA_FORMA','IN_RESERVA_VAGAS','IN_RESERVA_ETNICO','IN_RESERVA_DEFICIENCIA','IN_RESERVA_ENSINO_PUBLICO','IN_RESERVA_RENDA_FAMILIAR','IN_RESERVA_OUTRA','IN_FINANCIAMENTO_ESTUDANTIL','IN_FIN_REEMB_FIES','IN_FIN_REEMB_ESTADUAL','IN_FIN_REEMB_MUNICIPAL','IN_FIN_REEMB_PROG_IES','IN_FIN_REEMB_ENT_EXTERNA','IN_FIN_REEMB_OUTRA','IN_FIN_NAOREEMB_PROUNI_INTEGR','IN_FIN_NAOREEMB_PROUNI_PARCIAL','IN_FIN_NAOREEMB_ESTADUAL','IN_FIN_NAOREEMB_MUNICIPAL','IN_FIN_NAOREEMB_PROG_IES','IN_FIN_NAOREEMB_ENT_EXTERNA','IN_FIN_NAOREEMB_OUTRA','IN_APOIO_SOCIAL','IN_APOIO_ALIMENTACAO','IN_APOIO_BOLSA_PERMANENCIA','IN_APOIO_BOLSA_TRABALHO','IN_APOIO_MATERIAL_DIDATICO','IN_APOIO_MORADIA','IN_APOIO_TRANSPORTE','IN_ATIVIDADE_EXTRACURRICULAR','IN_COMPLEMENTAR_ESTAGIO','IN_COMPLEMENTAR_EXTENSAO','IN_COMPLEMENTAR_MONITORIA','IN_COMPLEMENTAR_PESQUISA','IN_BOLSA_ESTAGIO','IN_BOLSA_EXTENSAO','IN_BOLSA_MONITORIA','IN_BOLSA_PESQUISA','TP_ESCOLA_CONCLUSAO_ENS_MEDIO','IN_ALUNO_PARFOR','TP_SEMESTRE_CONCLUSAO','IN_MOBILIDADE_ACADEMICA','TP_MOBILIDADE_ACADEMICA','TP_MOBILIDADE_ACADEMICA_INTERN','IN_INGRESSO_TOTAL','IN_INGRESSO_VAGA_NOVA','IN_INGRESSO_PROCESSO_SELETIVO','NU_ANO_INGRESSO'])

#retira aqueles que estao cursando
df= df[df['TP_SITUACAO'] != 2]

df.update(df['TP_TURNO'].fillna(5))  # os em branco sao EAD vira 5 
df.update(df['TP_GRAU_ACADEMICO'].fillna(5))  #outros vira 5

#os dados missing viram nao tem reserva ou tipo de financiamento (0)
df.update(df['IN_RESERVA_ETNICO'].fillna(0))  
df.update(df['IN_RESERVA_DEFICIENCIA'].fillna(0))  
df.update(df['IN_RESERVA_ENSINO_PUBLICO'].fillna(0))
df.update(df['IN_RESERVA_RENDA_FAMILIAR'].fillna(0))
df.update(df['IN_RESERVA_OUTRA'].fillna(0))
df.update(df['IN_FIN_REEMB_FIES'].fillna(0))
df.update(df['IN_FIN_NAOREEMB_PROUNI_INTEGR'].fillna(0))
df.update(df['IN_FIN_NAOREEMB_PROUNI_PARCIAL'].fillna(0))
df.update(df['IN_FINANCIAMENTO_ESTUDANTIL'].fillna(0))

df.update(df['IN_APOIO_ALIMENTACAO'].fillna(0))
df.update(df['IN_APOIO_BOLSA_PERMANENCIA'].fillna(0))
df.update(df['IN_APOIO_BOLSA_TRABALHO'].fillna(0))
df.update(df['IN_APOIO_MATERIAL_DIDATICO'].fillna(0))
df.update(df['IN_APOIO_MORADIA'].fillna(0))
df.update(df['IN_APOIO_TRANSPORTE'].fillna(0))
df.update(df['IN_COMPLEMENTAR_ESTAGIO'].fillna(0))
df.update(df['IN_COMPLEMENTAR_EXTENSAO'].fillna(0))
df.update(df['IN_COMPLEMENTAR_MONITORIA'].fillna(0))
df.update(df['IN_COMPLEMENTAR_PESQUISA'].fillna(0))
df.update(df['IN_BOLSA_ESTAGIO'].fillna(0))
df.update(df['IN_BOLSA_EXTENSAO'].fillna(0))
df.update(df['IN_BOLSA_MONITORIA'].fillna(0))
df.update(df['IN_BOLSA_PESQUISA'].fillna(0)) 


df.update(df['IN_FIN_REEMB_ESTADUAL'].fillna(0)) 
df.update(df['IN_FIN_REEMB_MUNICIPAL'].fillna(0)) 
df.update(df['IN_FIN_REEMB_PROG_IES'].fillna(0)) 
df.update(df['IN_FIN_REEMB_ENT_EXTERNA'].fillna(0)) 
df.update(df['IN_FIN_REEMB_OUTRA'].fillna(0)) 
df.update(df['IN_FIN_NAOREEMB_ESTADUAL'].fillna(0)) 
df.update(df['IN_FIN_NAOREEMB_MUNICIPAL'].fillna(0)) 
df.update(df['IN_FIN_NAOREEMB_PROG_IES'].fillna(0)) 
df.update(df['IN_FIN_NAOREEMB_OUTRA'].fillna(0)) 
df.update(df['IN_FIN_NAOREEMB_ENT_EXTERNA'].fillna(0)) 



df.update(df['IN_ALUNO_PARFOR'].fillna(0)) 
df.update(df['IN_MOBILIDADE_ACADEMICA'].fillna(2)) 


#transforma campo semestre de ingresso 
def periodo(nome):
    nome = nome[4:5]
    return nome
df['DT_INGRESSO_CURSO'] = df['DT_INGRESSO_CURSO'].apply(periodo)




#------------ docente

df_docente = pd.read_csv('DM_DOCENTE.CSV',delimiter ='|',encoding='utf-8',usecols=['CO_IES','TP_ESCOLARIDADE','TP_REGIME_TRABALHO','TP_SEXO','TP_COR_RACA','IN_SUBSTITUTO','IN_ATUACAO_PESQUISA','IN_ATUACAO_POS_EAD','IN_ATUACAO_POS_PRESENCIAL','IN_ATUACAO_EAD','IN_ATUACAO_EXTENSAO'])


#dados ausentes alocados em uma classe
df_docente.update(df_docente['IN_SUBSTITUTO'].fillna(0))
df_docente.update(df_docente['IN_ATUACAO_PESQUISA'].fillna(0))
df_docente.update(df_docente['IN_ATUACAO_POS_EAD'].fillna(0))
df_docente.update(df_docente['IN_ATUACAO_POS_PRESENCIAL'].fillna(0))
df_docente.update(df_docente['IN_ATUACAO_EAD'].fillna(0))
df_docente.update(df_docente['IN_ATUACAO_EXTENSAO'].fillna(0))


#retira os professores com esse campo nao preenchido
df_docente= df_docente.dropna(subset=['TP_REGIME_TRABALHO'])



# codificacao binaria nas categorias de docentes
df_docente = pd.get_dummies(df_docente,columns=['TP_ESCOLARIDADE','TP_REGIME_TRABALHO','TP_SEXO','TP_COR_RACA','IN_SUBSTITUTO','IN_ATUACAO_PESQUISA','IN_ATUACAO_POS_EAD','IN_ATUACAO_POS_PRESENCIAL','IN_ATUACAO_EAD','IN_ATUACAO_EXTENSAO'])


#agrupa os dados por ies
df_docente = df_docente.groupby(by=['CO_IES'],as_index = False).sum()

#novos atributos em funcao
df_docente['Total_docente'] = df_docente['TP_SEXO_1']+df_docente['TP_SEXO_2']

df_docente['PER_MASC']=df_docente['TP_SEXO_2'] / df_docente['Total_docente']

df_docente['PER_GRADU']=df_docente['TP_ESCOLARIDADE_2'] / df_docente['Total_docente']

df_docente['PER_MESTRE']=df_docente['TP_ESCOLARIDADE_4'] / df_docente['Total_docente']

df_docente['PER_DOC']=df_docente['TP_ESCOLARIDADE_5'] / df_docente['Total_docente']

df_docente['PER_NEGROS_PARDOS']=(df_docente['TP_COR_RACA_2'] + df_docente['TP_COR_RACA_3']) / df_docente['Total_docente']

df_docente['PER_SUBSTITUTO']=df_docente['IN_SUBSTITUTO_1.0'] / df_docente['Total_docente']

df_docente['PER_PEQUISA']=df_docente['IN_ATUACAO_PESQUISA_1.0'] / df_docente['Total_docente']

df_docente['PER_TRAB_EAD']=df_docente['IN_ATUACAO_EAD_1.0'] / df_docente['Total_docente']

df_docente['PER_EXTENSAO']=df_docente['IN_ATUACAO_EXTENSAO_1.0'] / df_docente['Total_docente']

df_docente['PER_DEDIC_EXCL']=df_docente['TP_REGIME_TRABALHO_1.0'] / df_docente['Total_docente']

#retira atributos extras

df_docente=df_docente.drop(columns=['TP_ESCOLARIDADE_1','TP_ESCOLARIDADE_2','TP_ESCOLARIDADE_3','TP_ESCOLARIDADE_4','TP_ESCOLARIDADE_5','TP_REGIME_TRABALHO_1.0','TP_REGIME_TRABALHO_2.0','TP_REGIME_TRABALHO_3.0','TP_REGIME_TRABALHO_4.0','TP_SEXO_1','TP_SEXO_2','TP_COR_RACA_0','TP_COR_RACA_1','TP_COR_RACA_2','TP_COR_RACA_3','TP_COR_RACA_4','TP_COR_RACA_5','IN_SUBSTITUTO_0.0','IN_SUBSTITUTO_1.0','IN_ATUACAO_PESQUISA_0.0','IN_ATUACAO_PESQUISA_1.0','IN_ATUACAO_POS_EAD_0.0','IN_ATUACAO_POS_EAD_1.0','IN_ATUACAO_POS_PRESENCIAL_0.0','IN_ATUACAO_POS_PRESENCIAL_1.0','IN_ATUACAO_EAD_0.0','IN_ATUACAO_EAD_1.0','IN_ATUACAO_EXTENSAO_0.0','IN_ATUACAO_EXTENSAO_1.0'])


#-----------------ocde

df_ocde = pd.read_csv('TB_AUX_AREA_OCDE.CSV',delimiter ='|',encoding='latin-1',usecols=['CO_OCDE','NO_OCDE_AREA_GERAL','NO_OCDE_AREA_ESPECIFICA'])


MUNICIPIOS = pd.read_csv('MUNICIPIOS.csv',delimiter =';',encoding='latin-1')

# --------------------MERGE

df=pd.merge(df, df_docente, on=["CO_IES"], how="left")
df=pd.merge(df, df_ies, on=["CO_IES"], how="left")
df=pd.merge(df, df_ocde, on=["CO_OCDE"], how="left")
df=pd.merge(df, df_curso, on=["CO_CURSO"], how="left")
print("TAMANHO0",df.shape)



df=pd.merge(df, locali, on=["CO_MUNICIPIO"], how="left") #latitude longitude



print("TAMANHO1",df.shape)



#retira campos ja usados nas funcoes e ano do censo
df= df.drop(columns=['NU_ANO_CENSO','VL_RECEITA_PROPRIA','VL_RECEITA_TRANSFERENCIA','VL_RECEITA_OUTRA','VL_DESPESA_PESSOAL_DOCENTE','VL_DESPESA_PESSOAL_TECNICO','VL_DESPESA_PESSOAL_ENCARGO','VL_DESPESA_CUSTEIO','VL_DESPESA_INVESTIMENTO','VL_DESPESA_PESQUISA','VL_DESPESA_OUTRA'], axis =1)


#verificacao 

null=df.isnull().sum()
null.to_csv('missing.csv')

print("TAMANHO2",df.shape)

#k= df['TP_MODALIDADE_ENSINO']
#print("mod_ensino",k.value_counts())



#retira linha de alunos em cursos sem classificacao ocde 
df= df.dropna(subset=['CO_OCDE'])
df= df.dropna(subset=['IN_GRATUITO'])

print("TAMANHO3",df.shape)


#retira por motivo de muitos campos  ausentes
df= df.drop(columns=['CO_UF_NASCIMENTO','CO_MUNICIPIO_NASCIMENTO','CO_CURSO_POLO','IN_INGRESSO_OUTRO_TIPO_SELECAO','IN_INGRESSO_OUTRA_FORMA','TP_MOBILIDADE_ACADEMICA','TP_MOBILIDADE_ACADEMICA_INTERN','IN_INGRESSO_PROCESSO_SELETIVO'], axis =1)


#retira por inconsistencia de dados
df= df.drop(columns=['TP_SEMESTRE_CONCLUSAO','IN_INGRESSO_VAGA_NOVA','IN_MOBILIDADE_ACADEMICA','CO_PAIS_ORIGEM','TP_REFERENTE'], axis =1)

#retira por ser campos redundantes
df= df.drop(columns=['NO_CURSO','CO_MUNICIPIO','CO_IES','CO_CURSO','CO_OCDE','CO_UF'], axis =1)


# completa localizacao cursos ead 
df.update(df['codigo_uf'].fillna(0))
k= df['codigo_uf']
print("ufff",k.value_counts())


df.update(df['latitude'].fillna(0))
df.update(df['longitude'].fillna(0))


#transforma a classe em evadido(1) (trancado e desvinculado) e formado(0) (nao evadido)
legenda = {3: 1, 4: 1,6:0}
df['Evasao'] = df.TP_SITUACAO.replace(legenda)
df['Evasao'] = df.Evasao.astype('int')

#exclui os falecidos e transferido
df = df.loc[df['Evasao'] !=7]
df = df.loc[df['Evasao'] !=5]
df= df.drop(columns=['TP_SITUACAO'])

#campo de regiao
legenda_regiao = {0:6,12:1,27:2,13:1,16:1,29:2,23:2,53:3,32:5,52:3,21:2,31:5,50:3,51:3,15:1,25:2,26:2,22:2,41:4,33:5,24:2,11:1,14:1,43:4,42:4,28:2,35:5,17:1}
df['Regiao'] = df.codigo_uf.replace(legenda_regiao)
df['Regiao'] = df.Regiao.astype('int')

print('Regiao',df['Regiao'].value_counts())

#muda os codigos para numeros menores

df.loc[df.TP_DEFICIENCIA==9,'TP_DEFICIENCIA']=2
df.loc[df.TP_ESCOLA_CONCLUSAO_ENS_MEDIO==9,'TP_ESCOLA_CONCLUSAO_ENS_MEDIO']=2
df.loc[df.TP_MODALIDADE_ENSINO==2,'TP_MODALIDADE_ENSINO']=0
df.loc[df.TP_NIVEL_ACADEMICO==2,'TP_NIVEL_ACADEMICO']=0
df.loc[df.TP_SEXO==2,'TP_SEXO']=0
df.loc[df.TP_NACIONALIDADE==2,'TP_NACIONALIDADE']=0
df.loc[df.DT_INGRESSO_CURSO==1,'DT_INGRESSO_CURSO']=0
df.loc[df.DT_INGRESSO_CURSO==7,'DT_INGRESSO_CURSO']=1

print("TAMANHO antes de remover out",df.shape)


#grafico boxplot 

import matplotlib.pyplot as plt
plt.boxplot(df['NU_ANO_INGRESSO'])
plt.show()
plt.savefig('box_ano.png')
plt.close()

#loop retira outlier

def removeout(x):
  valor=x
  Q1= valor.quantile(.25)
  Q3= valor.quantile(.75)
  IIQ=Q3-Q1
  limite_inferior=Q1- 1.5 *IIQ
  limite_superior=Q3+ 1.5 *IIQ
  selecao=(valor >+limite_inferior) &(valor<=limite_superior)
  return selecao

lista_remover_out=['NU_ANO_INGRESSO']
#,'QT_CONCLUINTE_TOTAL','QT_MATRICULA_TOTAL','receitas','despesas']

print(df)
for i in lista_remover_out:
 df=df[removeout(df[i])]
 print(i)


plt.boxplot(df['NU_ANO_INGRESSO'])
plt.show()
plt.savefig('box_ano_out.png')
plt.close()

print("Tamanho apos remover out",df.shape)

#Dividir a base de dados em selecao e validacao

from sklearn.model_selection import StratifiedKFold

y=df['Evasao'] 

skf=StratifiedKFold(n_splits=2,shuffle =True)

for train_index, test_index in skf.split(df,y):
  print("TRAIN:", train_index, "TEST:", test_index)
  df_selecao, df_validacao = df.iloc[train_index], df.iloc[test_index]
  

df_selecao.to_csv('df_selecao.csv', index=False)

df_validacao.to_csv('df_validacao.csv', index=False)

k= df['NU_ANO_INGRESSO']
print("ano_ingresso",k.value_counts())


#analise exploratoria

#df_agrup= df[['codigo_uf','TP_MODALIDADE_ENSINO','TP_CATEGORIA_ADMINISTRATIVA','Evasao']]

#dfagrup= df_agrup.value_counts()
#dfagrup.to_csv('df_agrup.csv', index=False)