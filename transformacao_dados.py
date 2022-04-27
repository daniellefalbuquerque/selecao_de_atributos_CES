
#tranformacoes que impedem a analise mas facilitam o AM

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


from sklearn.preprocessing import LabelEncoder

def tranformacao(df):
  print("TAMANHO",df.shape)

#codificacao simples dados categoricos
  le =LabelEncoder()

  df['codigo_uf']=le.fit_transform(df['codigo_uf'])
  df['NO_IES']=le.fit_transform(df['NO_IES'])
  df['NO_OCDE_AREA_ESPECIFICA']=le.fit_transform(df['NO_OCDE_AREA_ESPECIFICA'])
  df['NO_OCDE_AREA_GERAL']=le.fit_transform(df['NO_OCDE_AREA_GERAL'])

  #normalizacao atrib numericos
  df[['QT_CARGA_HORARIA_TOTAL', 'NU_ANO_INGRESSO','NU_CARGA_HORARIA','QT_CONCLUINTE_TOTAL','QT_INGRESSO_TOTAL','QT_VAGA_TOTAL','QT_TEC_TOTAL','QT_MATRICULA_TOTAL','Total_docente','QT_TEC_TOTAL','QT_PERIODICO_ELETRONICO','QT_LIVRO_ELETRONICO','receitas','despesas','QT_INGRESSO_VAGA_NOVA','latitude','longitude']] = MinMaxScaler().fit_transform(df[['QT_CARGA_HORARIA_TOTAL','NU_ANO_INGRESSO','NU_CARGA_HORARIA','QT_CONCLUINTE_TOTAL','QT_INGRESSO_TOTAL','QT_VAGA_TOTAL','QT_TEC_TOTAL','QT_MATRICULA_TOTAL','Total_docente','QT_TEC_TOTAL','QT_PERIODICO_ELETRONICO','QT_LIVRO_ELETRONICO','receitas','despesas','QT_INGRESSO_VAGA_NOVA','latitude','longitude']])


  #normalizacao atrib categoricos
  df[['TP_CATEGORIA_ADMINISTRATIVA','TP_ORGANIZACAO_ACADEMICA','TP_TURNO','TP_GRAU_ACADEMICO','TP_COR_RACA','NU_IDADE','TP_DEFICIENCIA','codigo_uf','TP_ESCOLA_CONCLUSAO_ENS_MEDIO','NO_OCDE_AREA_GERAL','NO_OCDE_AREA_ESPECIFICA','NO_IES','Regiao']] = MinMaxScaler().fit_transform(df[['TP_CATEGORIA_ADMINISTRATIVA','TP_ORGANIZACAO_ACADEMICA','TP_TURNO','TP_GRAU_ACADEMICO','TP_COR_RACA','NU_IDADE','TP_DEFICIENCIA','codigo_uf','TP_ESCOLA_CONCLUSAO_ENS_MEDIO','NO_OCDE_AREA_GERAL','NO_OCDE_AREA_ESPECIFICA','NO_IES','Regiao']])


  #arredonda a casas decimais 
  df=df.round(4)
  print("TAMANHO",df.shape)
  return df

df_selecao = pd.read_csv('df_selecao.csv',delimiter =',',encoding='utf-8')
base_selecao_v2=tranformacao(df_selecao)
base_selecao_v2.to_csv('base_selecao_v2.csv', index=False)

df_validacao = pd.read_csv('df_validacao.csv',delimiter =',',encoding='utf-8')
base_validacao_v2=tranformacao(df_validacao)
base_validacao_v2.to_csv('base_validacao_v2.csv', index=False)




