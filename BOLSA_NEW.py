#!/usr/bin/env python
# coding: utf-8

# In[1]:


################################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR,                  ENG. CIVIL / PROF (UFCAT)
# ANTOVER PANAZZOLO SARMENTO,                       ENG. AGRICOLA / PROF (UFCAT)
# MARIA JOSÉ PEREIRA DANTAS,                                 MAT. / PROF (UFCAT)
# EULLER SANTOS MIRANDA,                                           COMP. (UFCAT)
# JOÃO COELHO ESTRELA,                                        ENG. MINAS (UFCAT)   
# FABRICIO NUNES MOLTAVÃO,                                      ENG. CIVIL (UEG)         
# DANILO MILHOMEM,                                        ENG. PRODUÇÃO (PUC-GO)                             
################################################################################

################################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIOTECA DE FUNÇÕES PARA TRATAMENTO INICIAL DE DADOS DA BOLSA DE VALORES 
# BRASILEIRA DESENVOLVIDA PELO GRUPO DE PESQUISAS E ESTUDOS EM ENGENHARIA (GPEE)
################################################################################


################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE


# In[2]:


def DADOS_BOLSA_PRECO_AJUSTADO(DADOS):
    """
    Esta função recolhe dados da bolsa de valores empregando a biblioteca Yahoo Finance.

    Entrada:
    DADOS          | Dados necessários para rodar a biblioteca  | Py dictionary
                   |   Etiquetas do dicionário:                 |
                   |   'ID ACAO'      = Ticker da ação          | Py list
                   |   'DATA INICIAL' = Data inicial dos dados  | String
                   |   'DATA FINAL '  = Data final dos dados    | String
    Saída:
    DADOS_BOLSA    | Dados da bolsa de valores Volume, Preço    | Py dictionary
                   | Máximo, Mínimo, Abertura, Fechamento e     |
                   | Fechamento ajustado                        |
    DADOS_RETORNO  | Retorno do ativo                           | Py dictionary   
    DADOS_COV      | Matriz de covariância dos ativos           | Py dictionary 
    """
    # Recolhendo os dados via biblioteca yfinance
    ACOES = DADOS['ID ACAO']
    DATA_INICIAL = DADOS['DATA INICIAL']
    DATA_FINAL = DADOS['DATA FINAL']
    # Todos os dados
    DADOS_BOLSA = yf.download(ACOES, start = DATA_INICIAL, end = DATA_FINAL)
    DADOS_BOLSA = DADOS_BOLSA.dropna()
    DADOS_PRECO = DADOS_BOLSA['Adj Close']
    DADOS_RETORNO = DADOS_PRECO.pct_change()
    DADOS_RETORNO.columns = ACOES
    DADOS_COV = DADOS_PRECO.cov()
    return DADOS_BOLSA, DADOS_RETORNO, DADOS_COV

def BOLSA_PLOT_001(DADOS, OPCOES_GRAF):
    """
    Plotagem do mapa de calor da matriz de covariância.

    Entrada:  
    DADOS       | Matriz de covariância                                    | Py dictionary    
    OPCOES_GRAF | Opções gráficas                                          | Py Dictionary
                |  Dictionary tags                                         |
                |    'NAME'          == Filename output file               | String 
                |    'WIDTH'         == Width figure                       | Float
                |    'HEIGHT         == Height figure                      | Float
                |    'EXTENSION'     == Extension output file              | String 
                |    'DPI'           == Dots Per Inch - Image quality      | Integer   
                |    'COLOR OF'      == OF line color                      | String
                |    'ANNOT'         ==                                    | Boolean
                |    'LINEWIDTHS'    == espaço entre o valores             | Integer
                |    'FMT'           == Código de formatação de string     | String
      
    Saida:
    The image is saved in the current directory 
    """
    ANNOT = OPCOES_GRAF['ANNOT']
    LINEWIDTHS = OPCOES_GRAF['LINEWIDTHS']
    FMT = OPCOES_GRAF['FMT']
    sns.heatmap(DADOS, annot = ANNOT, linewidths = LINEWIDTHS , fmt = FMT)

def BOLSA_PLOT_002(DADOS):
    
    ID_ACAO = DADOS['ID ACAO']
    DADOS_PRECOS = DADOS['PRECOS']
    DATA_INICIAL = DADOS['DATA INICIAL']
    DATA_FINAL = DADOS['DATA FINAL']
    PERIODO_MOVEL = DADOS['PERIODO MEDIA MOVEL']
    
    plt.figure(figsize=(24,12))
    DADOS_PRECOS[ID_ACAO].loc[DATA_INICIAL : DATA_FINAL].rolling(window=PERIODO_MOVEL).mean().plot(label='Media movel:'+str(PERIODO_MOVEL)+' dias')
    DADOS_PRECOS[ID_ACAO].loc[DATA_INICIAL : DATA_FINAL].plot(label= str(ID_ACAO + ' ADJ CLOSE'))
    plt.legend()


def FO_MARKOWITZ(X, DADOS_COV, DADOS_RETORNO, LAMBDA):
    """
    Esta função determina o valor da função objetivo do problema de portifólio financeiro
    proposto por Henry Markowitz.
    
    Entrada:
    X              | Proporção dos ativos establecidos para a carteira indivíduo I   | Py list[D]
    DADOS_COV      | Matriz de covariância dos ativos do portifólio                  | Py dictionary 
    DADOS_RETORNO  | Retorno do ativos ativos do portifólio                          | Py dictionary   
    LAMBDA         | Aversão ao risco para o portifólio                              | Float
    
    Saída:
    OF             | Valor de função objetivo para o indivíduo  I                    | Float
    """
    # Variância do portifólio
    D = len(X)
    MATRIZ_COV = DADOS_COV.values
    VARIANCIA = []
    for I_COUNT in range(D):
        if X[I_COUNT] != 0:
            for J_COUNT in range(D):
                if X[J_COUNT] != 0:
                    variancia = X[I_COUNT] * X[J_COUNT] * MATRIZ_COV[I_COUNT][J_COUNT]
                    VARIANCIA.append(variancia)
    VARIANCIA_PORTIFOLIO = sum(VARIANCIA)    

    # Retorno do portifólio
    RETORNO_MEDIO = DADOS_RETORNO.mean()
    RETORNO = 0
    for K_COUNT in range(D):
        if X[K_COUNT] != 0:
            RETORNO += X[K_COUNT] * RETORNO_MEDIO[K_COUNT]

    # Função objetivo com parametro de aversão ao risco: fronteira eficiente
    FO = LAMBDA * VARIANCIA_PORTIFOLIO - (1 - LAMBDA) * RETORNO
    return FO

