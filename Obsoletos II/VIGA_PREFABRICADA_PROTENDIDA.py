################################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR,                  ENG. CIVIL / PROF (UFCAT)
# SYLVIA REGINA MESQUISTA DE ALMEIDA,                 ENG. CIVIL / PROF (UFG-GO)
# MATHEUS HENRIQUE MORATO DE MORAES,                          ENG. CIVIL (UFCAT)
# GERALDO MAGELA FILHO,                                       ENG. CIVIL (UFCAT)
# GUSTAVO GONÇALVES COSTA,                                    ENG. CIVIL (UFCAT)
################################################################################

################################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIOTECA DE DIMENSIONAMENTO DE VIGAS PRÉ-FABRICADAS E PROTENDIDAS DESENVOL-
# VIDA PELO GRUPO DE PESQUISA E ESTUDOS EM ENGENHARIA (GPEE)
################################################################################

################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import numpy as np
import math

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE

def PROP_GEOMETRICA_VIGA_I(H, B_FS, B_FI, B_W, H_S, H_I, H_SI, H_II):
    """
    Esta função determina as propriedades geométricas de uma seção I com abas inclinadas.

    Entrada:
    H         | Altura da viga                                     | m    | float
    B_FS      | Base de mesa superior da viga                      | m    | float
    B_FI      | Base de mesa inferior da viga                      | m    | float
    B_W       | Base de alma da viga                               | m    | float
    H_S       | Altura de mesa superior da viga                    | m    | float
    H_I       | Altura de mesa inferior da viga                    | m    | float
    H_SI      | Altura inclinada de mesa superior da viga          | m    | float
    H_II      | Altura inclinada de mesa inferior da viga          | m    | float

    Saída:
    A_C       | Área da  seção transversal da viga                 | m²   | float
    I_C       | Inércia da viga                                    | m^4  | float
    Y_SUP     | Ordenada da fibra superior                         | m    | float 
    Y_INF     | Ordenada da fibra inferior                         | m    | float
    W_SUP     | Modulo de resistência superior                     | m³   | float
    W_INF     | Modulo de resistência inferior                     | m³   | float
    """
    A_1 = B_W * H
    A_2 = (B_FS - B_W) * H_S
    A_3 = ((B_FS - B_W) * H_SI) / 2
    A_4 = (B_FI - B_W) * H_I
    A_5 = ((B_FI - B_W) * H_II)/2
    A_C = A_1 + A_2 + A_3 + A_4 + A_5  
    Y_CG = (A_1 * H / 2 + A_2 * (H - H_S / 2) + A_3 * (H - H_S - H_SI / 3) + A_4 * H_I / 2 + A_5 * (H_I + H_II / 3)) /(A_C)
    I_1 = (B_W * H**3) / 12 + A_1 * (H / 2 - Y_CG)**2 
    I_2 = ((B_FS - B_W)* H_S**3) / 12 + A_2 * (H - H_S/2 - Y_CG)**2 
    I_3 = ((B_FS - B_W)* H_SI**3) / 36 + A_3 * (H - H_S - H_SI / 3 - Y_CG)**2 
    I_4 = ((B_FI - B_W)* H_I**3) / 12 + A_4 * (Y_CG - H_I / 2)**2 
    I_5 = ((B_FI - B_W)* H_II**3) / 36 + A_5 * (Y_CG - H_I - H_II / 3)**2 
    I_C = I_1 + I_2 + I_3 + I_4 + I_5
    Y_SUP = H - Y_CG 
    Y_INF = Y_CG
    W_SUP = I_C / Y_SUP 
    W_INF = I_C / Y_INF     
    return A_C, I_C, Y_SUP, Y_INF, W_SUP, W_INF

def PROP_GEOMETRICA_VIGA_RET(B_W, H):
    """
    Esta função determina as propriedades geométricas de uma seção retangular.

    Entrada:
    B_W       | Largura da viga                        | m    | float 
    H         | Altura da viga                         | m    | float

    Saída:
    A_C       | Área da  seção transversal da viga     | m²   | float
    I_C       | Inércia da viga                        | m^4  | float
    Y_SUP     | Ordenada da fibra superior             | m    | float 
    Y_INF     | Ordenada da fibra inferior             | m    | float
    W_SUP     | Modulo de resistência superior         | m³   | float
    W_INF     | Modulo de resistência inferior         | m³   | float
    """
    A_C = B_W * H 
    I_C = (B_W * H ** 3) / 12
    Y_SUP = H / 2 
    Y_INF = H / 2
    W_SUP = I_C / Y_SUP 
    W_INF = I_C / Y_INF 
    return A_C, I_C, Y_SUP, Y_INF, W_SUP, W_INF

def FATOR_BETA1(TEMPO, CIMENTO):
    """
    Esta função calcula o valor de BETA_1 que representa a função de crescimento da resistência do cimento.

    Entrada:
    TEMPO       | Tempo                                          | dias  | float
    CIMENTO     | Cimento utilizado                              |       | string    
                |   'CP1' - Cimento portland 1                   |       | 
                |   'CP2' - Cimento portland 2                   |       |              
                |   'CP3' - Cimento portland 3                   |       |
                |   'CP4' - Cimento portland 4                   |       | 
                |   'CP5' - Cimento portland 5                   |       | 
    
    Saída:
    BETA_1      | Parâmetro de crescimento da resistência        |       | float   
    """
    if TEMPO < 28 :
        if CIMENTO == 'CP1' or CIMENTO == 'CP2':
              S = 0.25  
        elif CIMENTO == 'CP3' or CIMENTO == 'CP4':
              S = 0.38  
        elif CIMENTO == 'CP5':
              S = 0.20  
        BETA_1 = np.exp(S * (1 - (28 / TEMPO) ** 0.50))
    else :
        BETA_1 = 1
    return BETA_1

def MODULO_ELASTICIDADE_CONCRETO(AGREGADO, F_CK, F_CKJ):
    """
    Esta função calcula os módulos de elasticidade do concreto.  

    Entrada:
    AGREGADO    | Tipo de agragado usado no traço do cimento       |        | string    
                |   'BAS' - Agregado de Basalto                    |        | 
                |   'GRA' - Agregado de Granito                    |        |              
                |   'CAL' - Agregado de Calcário                   |        |
                |   'ARE' - Agregado de Arenito                    |        | 
    F_CK        | Resistência característica à compressão          | kN/m²  | float   
    F_CKJ       | Resistência característica à compressão idade J  | kN/m²  | float
    
    Saída:
    E_CIJ       | Módulo de elasticidade tangente                  | kN/m²  | float
    E_CSJ       | Módulo de elasticidade do secante                | kN/m²  | float   
    """
    # Determinação do módulo tangente E_CI idade T
    if AGREGADO == 'BAS':         
        ALFA_E = 1.2
    elif AGREGADO == 'GRA':         
        ALFA_E = 1.0
    elif AGREGADO == 'CAL':       
        ALFA_E = 0.9
    elif AGREGADO == 'ARE':       
        ALFA_E = 0.7
    F_CK /= 1E3
    if F_CK <= 50:        
        E_CI = ALFA_E * 5600 * np.sqrt(F_CK)
    elif F_CK > 50:   
        E_CI = 21.5 * (10 ** 3) * ALFA_E * (F_CK / 10 + 1.25) ** (1 / 3)
    ALFA_I = 0.8 + 0.2 * F_CK / 80
    if ALFA_I > 1:        
        ALFA_I = 1
    # Determinação do módulo secante E_CS idade T
    E_CS = E_CI * ALFA_I
    if F_CK <= 45 :
        F_CK *= 1E3
        E_CIJ = E_CI * (F_CKJ / F_CK) ** 0.5  
    elif  F_CK > 45 : 
        F_CK *= 1E3
        E_CIJ = E_CI * (F_CKJ / F_CK) ** 0.3  
    E_CSJ = E_CIJ * ALFA_I
    E_CIJ *= 1E3 
    E_CSJ *= 1E3 
    return E_CIJ, E_CSJ

def PROP_MATERIAL(F_CK, TEMPO, CIMENTO, AGREGADO):
    """
    Esta função determina propriedades do concreto em uma idade TEMPO.
    
    Entrada:
    F_CK        | Resistência característica à compressão                | kN/m²  | float   
    TEMPO       | Tempo                                                  | dias   | float
    CIMENTO     | Cimento utilizado                                      |        | string    
                |   'CP1' - Cimento portland 1                           |        | 
                |   'CP2' - Cimento portland 2                           |        |              
                |   'CP3' - Cimento portland 3                           |        |
                |   'CP4' - Cimento portland 4                           |        | 
                |   'CP5' - Cimento portland 5                           |        | 
    AGREGADO    | Tipo de agragado usado no traço do cimento             |        | string    
                |   'BAS' - Agregado de Basalto                          |        | 
                |   'GRA' - Agregado de Granito                          |        |              
                |   'CAL' - Agregado de Calcário                         |        |
                |   'ARE' - Agregado de Arenito                          |        | 
    
    Saída:
    F_CKJ       | Resistência característica à compressão idade J        | kN/m²  | float
    F_CTMJ      | Resistência média caracteristica a tração idade J      | kN/m²  | float
    F_CTKINFJ   | Resistência média caracteristica a tração inf idade J  | kN/m²  | float
    F_CTKSUPJ   | Resistência média caracteristica a tração sup idade J  | kN/m²  | float
    E_CIJ       | Módulo de elasticidade tangente                        | kN/m²  | float
    E_CSJ       | Módulo de elasticidade do secante                      | kN/m²  | float      
    """
    # Propriedades em situação de compressão F_C idade TEMPO em dias
    BETA_1 = FATOR_BETA1(TEMPO, CIMENTO)
    F_CKJ = F_CK * BETA_1
    F_CKJ /= 1E3
    F_CK /= 1E3
    if F_CKJ < 21 :
        F_CKJ = 21
    # Propriedades em situação de tração F_CT idade TEMPO em dias
    if F_CK <= 50:
          F_CTMJ = 0.3 * F_CKJ ** (2/3)
    elif F_CK > 50:
          F_CTMJ = 2.12 * np.log(1 + 0.11 * F_CKJ)
    F_CTMJ *= 1E3
    F_CTKINFJ = 0.7 * F_CTMJ 
    F_CTKSUPJ = 1.3 * F_CTMJ
    # Módulo de elasticidade do concreto
    F_CKJ *= 1E3
    F_CK *= 1E3
    [E_CIJ, E_CSJ] = MODULO_ELASTICIDADE_CONCRETO(AGREGADO, F_CK, F_CKJ)
    return  F_CKJ, F_CTMJ, F_CTKINFJ, F_CTKSUPJ, E_CIJ, E_CSJ 

def TENSAO_INICIAL(TIPO_PROT, TIPO_ACO, F_PK, F_YK):
    """
    Esta função determina a tensão inicial de protensão e a carga inicial de protensão.

    Entrada:
    TIPO_PROT  | Protensão utilizada                                  |       | string    
               |   'PRE' - Peça pré tracionada                        |       | 
               |   'POS' - Peça pós tracionada                        |       |  
    TIPO_ACO   | Tipo de aço                                          |       | string
               |   'RN' - Relaxação normal                            |       |
               |   'RB' - Relaxação baixa                             |       |
    F_PK       | Tensão última característica do aço                  | kN/m² | float
    F_YK       | Tensão de escoamento característica do aço           | kN/m² | float   

    Saída:
    SIGMA_PIT0 | Tensão inicial de protensão                          | kN/m² | float
    
    """
    if TIPO_PROT == 'PRE':
        if TIPO_ACO == 'RN':
            SIGMA_PIT0 = min(0.77 * F_PK, 0.90 * F_YK)
        elif TIPO_ACO == 'RB':
            SIGMA_PIT0 = min(0.77 * F_PK, 0.85 * F_YK)       
    elif TIPO_PROT == 'POS':
        if TIPO_ACO == 'RN':
            SIGMA_PIT0 = min(0.74 * F_PK, 0.87 * F_YK)
        elif TIPO_ACO == 'RB':
            SIGMA_PIT0 = min(0.74 * F_PK, 0.82 * F_YK)
    return SIGMA_PIT0

def COMPRIMENTO_TRANSFERENCIA(PHI_L, F_YK, F_CTKINFJ, ETA_1, ETA_2, SIGMA_PI, H):
    """
    Esta função calcula o comprimento de tranferência da armadura L_P.

    Entrada:
    PHI_L      | Diâmetro da armadura                                   | m      | float
    F_YK       | Tensão de escoamento característica do aço             | kN/m²  | float
    F_CTKINFJ  | Resistência média caracteristica a tração inf idade J  | kN/m²  | float
    ETA_1      | Fatores de aderência                                   |        | float
    ETA_2      | Fatores de aderência                                   |        | float
    SIGMA_PI   | Tensão de protensão na etapa desejada                  | kN/m²  | float
    H          | Altura da peça de concreto                             | m      | float 

    Saída:
    L_P        |  Comprimento de transferência                          | m      | float
    """ 
    F_YD = F_YK / 1.15
    F_CTD = F_CTKINFJ / 1.4
    F_BPD = ETA_1 * ETA_2 * F_CTD
    # Comprimento de ancoragem básico para cordoalhas
    L_BP = (7 * PHI_L * F_YD) / (36 * F_BPD)
    # Comprimento básico de transferência para cordoalhas não gradual
    L_BPT = (0.625 * L_BP ) * (SIGMA_PI/ F_YD)
    AUXL_P = np.sqrt(H ** 2 + (0.6 * L_BPT) ** 2) 
    L_P = max(AUXL_P, L_BPT)
    return L_P

def ESFORCOS(Q, L, L_P):
    """
    Esta função determina os esforços atuantes na viga biapoiada.
    
    Entrada:
    Q           | Carga lineramente distribuida      | kN/m  | float
    L           | Comprimento da viga                | m     | float
    L_P         | Comprimento de transferência       | m     | float

    Saída:
    M          | Momento atuante no meio da viga    | kNm    | float
    M_AP       | Momento atuante no L_P             | kNm    | float
    V          | Cortante atuante no apoio da viga  | kN     | float
    """
    # Momento no meio do vão
    M_MV = Q * (L ** 2) / 8
    # Momento no apoio nas condições iniciais e finais
    M_AP = (Q * L / 2) * L_P - (Q * L_P / 1) * (L_P / 2)
    # Cortanto nos apoios
    V_AP = Q * L / 2 
    return M_MV, M_AP, V_AP 

def ESFORCOS_TRANSITORIOS(Q, L, CHI):
    """
    Esta função determina os esforços atuantes na viga biapoiada considerando
    as situações de içamento e armazenamento.
    
    Entrada:
    Q           | Carga lineramente distribuida                               | kN/m  | float
    L           | Comprimento da viga                                         | m     | float
    CHI         | Proporção de L para a posição dos dispositivos de içamento  |       | float

    Saída:
    M_POS       | Momento positivo atuante no meio da viga                    | kNm   | float
    M_NEG       | Momento negativo atuante no apoio da viga                   | kNm   | float
    """
    # Momento no meio do vão
    M_POS = (Q * (L ** 2) / 8) * (1 - 4 * CHI)
    M_NEG = Q * ((CHI * L) ** 2) / 2
    return M_POS, M_NEG

def TENSOES_NORMAIS(P_I, A_C, E_P, W_INF, W_SUP, DELTA_P, DELTA_G1, DELTA_G2, DELTA_G3, DELTA_Q1, DELTA_Q2, PSI_Q1, M_G1, M_G2, M_G3, M_Q1, M_Q2):
    """
    Esta função determina a tensão normal nos bordos inferior e superior da peça.
    
    Entrada:
    P_I         | Carga de protensão considerando as perdas         | kN      | float
    A_C         | Área da  seção transversal da viga                | m²      | float
    E_P         | Excentricidade de protensão                       | m       | float 
    W_SUP       | Modulo de resistência superior                    | m³      | float
    W_INF       | Modulo de resistência inferior                    | m³      | float
    DELTA_      | Coeficientes parciais de segurança (G,Q,P)        |         | float
    PSI_Q1      | Coeficiente parcial de segurança carga Q_1        |         | float
    M_          | Momentos caracteristicos da peça (G,Q)            | kNm     | float  
        
    Saída:
    SIGMA_INF   | Tensão normal fibra inferior                      | kN/m²   | float
    SIGMA_SUP   | Tensão normal fibra superior                      | kN/m²   | float
    """
    # Tensão normal fibras inferiores
    # Parcela da protensão
    AUX_0 = P_I / A_C
    AUX_1 = P_I * E_P / W_INF
    AUX_PINF =  DELTA_P * (AUX_0 + AUX_1) 
    # Parcela da carga permanente de PP
    AUX_G1INF = -1 * DELTA_G1 * M_G1 / W_INF 
    # Parcela da carga permanente da capa
    AUX_G2INF = -1 * DELTA_G2 * M_G2 / W_INF
    # Parcela da carga permanente do revestimento
    AUX_G3INF = -1 * DELTA_G3 * M_G3 / W_INF
    # Parcela da carga acidental de utilização
    AUX_Q1INF = -1 * DELTA_Q1 * PSI_Q1 * M_Q1 / W_INF
    # Parcela da carga acidental de montagem da peça
    AUX_Q2INF = -1 * DELTA_Q2 * M_Q2 / W_INF
    # Total para parte inferior
    SIGMA_INF = AUX_PINF + (AUX_G1INF + AUX_G2INF + AUX_G3INF ) + (AUX_Q1INF + AUX_Q2INF)
    # Tensão normal fibras Superior
    # Parcela da protensão
    AUX_PSUP =  DELTA_P * (P_I / A_C - P_I * E_P / W_SUP) 
    # Parcela da carga permanente de PP
    AUX_G1SUP = 1 * DELTA_G1 * M_G1 / W_SUP 
    # Parcela da carga permanente da capa
    AUX_G2SUP = 1 * DELTA_G2 * M_G2 / W_SUP
    # Parcela da carga permanente do revestimento
    AUX_G3SUP = 1 * DELTA_G3 * M_G3 / W_SUP
    # Parcela da carga acidental de utilização
    AUX_Q1SUP = 1 * DELTA_Q1 * PSI_Q1 * M_Q1 / W_SUP
    # Parcela da carga acidental de montagem da peça
    AUX_Q2SUP = 1 * DELTA_Q2 * M_Q2 / W_SUP
    # Total para parte inferior
    SIGMA_SUP = AUX_PSUP + (AUX_G1SUP + AUX_G2SUP + AUX_G3SUP) + (AUX_Q1SUP + AUX_Q2SUP)
    return SIGMA_INF, SIGMA_SUP

def VERIFICA_TENSAO_NORMAL_ATO_PROTENSÃO(SIGMA_INF, SIGMA_SUP, SIGMA_TRACMAX, SIGMA_COMPMAX):
    """
    Esta função verifica a restrição de tensão normal em peças estruturais conforme
    disposto na seção 17.2.4.3.2 da NBR 6118.
    
    Entrada:
    SIGMA_INF       | Tensão normal fibra inferior                      | kN/m²   | float
    SIGMA_SUP       | Tensão normal fibra superior                      | kN/m²   | float
    SIGMA_TRACMAX   | Tensão normal máxima na tração                    | kN/m²   | float
    SIGMA_COMPMAX   | Tensão normal máxima na compressão                | kN/m²   | float

    Saída:
    G_0             | Valor da restrição análise bordo inferior         |         | float
    G_1             | Valor da restrição análise bordo superior         |         | float
    """
    # Análise bordo inferior
    if SIGMA_INF >= 0:
        SIGMA_MAX = SIGMA_COMPMAX
        SIGMA_0 = SIGMA_INF
    else:
        SIGMA_MAX = SIGMA_TRACMAX
        SIGMA_0 = np.abs(SIGMA_INF)
    G_0 = (SIGMA_0 / SIGMA_MAX) - 1 
    # Análise bordo superior
    if SIGMA_SUP >= 0:
        SIGMA_MAX = SIGMA_COMPMAX
        SIGMA_1 = SIGMA_SUP
    else:
        SIGMA_MAX = SIGMA_TRACMAX
        SIGMA_1 = np.abs(SIGMA_SUP)
    G_1 = (SIGMA_1 / SIGMA_MAX) - 1 
    return G_0, G_1   

def AREA_ACO_TRANSVERSAL_MODELO_I(ALPHA, P_I, V_SD, F_CTKINFJ, B_W, D, TIPO_CONCRETO, W_INF, A_C, E_P, M_SDMAX, F_CTMJ, F_YWK):
    """
    Esta função verifica o valor da área de aço necessária para a peça de concreto.

    Entrada:
    ALPHA          | Inclinação do cabo protendido                          | graus | float
    P_I            | Carga de protensão considerando as perdas              | kN    | float    
    V_SD           | Cortante de cálculo                                    | kN    | float
    F_CTKINFJ      | Resistência média caracteristica a tração inf idade J  | kN/m² | float
    B_W            | Largura da viga                                        | m     | float  
    D              | Altura útil da seção                                   | m     | float
    TIPO_CONCRETO  | Defina se é concreto protendido ou armado              |       | string
                   |       'CP' - Concreto protendido                       |       |
                   |       'CA' - Concreto armado                           |       |
    W_INF          | Modulo de resistência inferior                         | m³    | float
    A_C            | Área da  seção transversal da viga                     | m²    | float
    E_P            | Excentricidade de protensão                            | m²    | float 
    M_SDMAX        | Momento de cálculo máximo                              | kN.m  | float
    F_CTMJ         | Resistência média caracteristica a tração idade J      | kN/m² | float
    F_YWK          | Resistência característica do aço do estribo           | kN/m² | float

    Saída:
    V_C            | Resitência ao cisalhamento do concreto                 | kN    | float
    V_SW           | Resitência ao cisalhamento da armadura                 | kN    | float    
    A_SW           | Área de aço para cisalhamento                          | m²/m  | float
    """
    # Contribuição do concreto na resistência
    F_CTD = F_CTKINFJ / 1.40
    V_C0 = 0.6 * F_CTD * B_W * D
    if TIPO_CONCRETO == 'CA':
        V_C = V_C0
    elif TIPO_CONCRETO == 'CP':
        # Correção do cisalhamento em função do esforço de protensão P_I
        ALPHA *= np.pi / 180
        N_P = P_I *np.cos(ALPHA)
        V_P = P_I *np.sin(ALPHA)
        V_SD -= 0.90 * V_P 
        AUX = N_P / A_C + (N_P * E_P) / W_INF
        M_0 = 0.90 * W_INF * AUX
        # Cálculo V_C
        V_CCALC = V_C0 * (1 + M_0 / M_SDMAX)
        if V_CCALC > (2 * V_C0):
            V_C = 2 * V_C0
        else:
            V_C = V_CCALC
    # Determinação da armadura
    if V_C >= V_SD:
        V_SW = 0
        F_CTMJ /= 1E3
        F_CTMJ /= 10
        B_W *= 1E2
        F_YWK /= 1E3
        F_YWK /= 10
        A_SW = (20 * F_CTMJ * B_W) / F_YWK
        A_SW /= 1E4
    else:
        ALPHA_EST = np.pi / 2
        V_SW = V_SD - V_C
        F_YWK /= 1E3
        F_YWK /= 10        
        F_YWD = F_YWK / 1.15
        D *= 1E2
        AUX = 0.90 * D * F_YWD * (np.sin(ALPHA_EST) + np.cos(ALPHA_EST))
        A_SW = V_SW / AUX
        A_SW /= 1 / 1E2
        A_SW /= 1E4
    return V_C, V_SW, A_SW 

def RESISTENCIA_BIELA_COMPRIMIDA(F_CK, B_W, D):
    """
    Esta função verifica o valor da resistência da biela comprimida V_RD2.

    Entrada:
    F_CK        | Resistência característica à compressão         | kN/m² | float
    B_W         | Largura da viga                                 | m     | float  
    D           | Altura útil da seção                            | m     | float
    
    Saída:
    V_RD2       | Resitência da biela comprimida                  | kN    | float 
    """
    # Força resistente da biela de compressão
    F_CK /= 1E3 
    ALFA_V2 = (1 - (F_CK / 250))
    F_CK *= 1E3 
    F_CD = F_CK / 1.40
    V_RD2 = 0.27 * ALFA_V2 * F_CD * B_W * D
    return V_RD2

def VERIFICA_BIELA_COMPRIMIDA(V_SD, V_MAX):
    """
    Esta função verifica a restrição do esforço na biela de compressão.
    
    Entrada:
    V_SD       | Cortante de cálculo                               | kN    | float
    V_MAX      | Cortante máximo permitido na biela de compressão  | kN    | float

    Saída:
    G_0        | Valor da restrição analisando o cisalhamento      |       | float
    """
    G_0 = (V_SD / V_MAX) - 1 
    return G_0  

def TENSAO_ACO(E_SCP, EPSILON, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a tensão da armadura de protensão a partir de 
    um valor de deformação.

    Entrada:
    E_SCP       | Módulo de elasticidade do aço protendido        | kN/m² | float
    EPSILON     | Deformação correspondente a tensão SIGMA 
    desejada                                                      |       | float
    EPSILON_P   | Deformação última do aço                        |       | float
    EPSILON_Y   | Deformação escoamento do aço                    |       | float
    F_Y         | Tensão de escoamento do aço                     | kN/m² | float
    F_P         | Tensão última do aço                            | kN/m² | float
    
    Saída:
    SIGMA       | Tensão correspondente a deformação Deformação   | kN/m² | float
    """
    # Determinação da tensão SIGMA correspodente a deformação EPSILON
    if EPSILON < (F_Y / E_SCP) :
        SIGMA = E_SCP * EPSILON
    elif EPSILON >= (F_Y / E_SCP):
        AUX = (F_P - F_Y) / (EPSILON_P - EPSILON_Y)
        SIGMA = F_Y + AUX * (EPSILON - EPSILON_Y)
    return SIGMA

def DEFORMACAO_ACO(E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a deformação da armadura de protensão a partir de 
    um valor de tensão.

    Entrada:
    E_SCP       | Módulo de elasticidade do aço protendido        | kN/m² | float
    SIGMA       | Tensão correspondente a tensão EPSILON desejada | kN/m² | float
    EPSILON_P   | Deformação última do aço                        |       | float
    EPSILON_Y   | Deformação escoamento do aço                    |       | float
    F_Y         | Tensão de escoamento do aço                     | kN/m² | float
    F_P         | Tensão última do aço                            | kN/m² | float
    
    Saída:
    EPSILON     | Deformação correspondente a tensão SIGMA        |       | float
    """
    # Determinação da deformação EPSILON correspodente a tensão SIGMA
    if (SIGMA / E_SCP) < (F_Y / E_SCP):      
        EPSILON = SIGMA / E_SCP
    elif (SIGMA / E_SCP) >= (F_Y / E_SCP):
        AUX = (F_P - F_Y) / (EPSILON_P - EPSILON_Y)
        EPSILON = (SIGMA - F_Y) / AUX + EPSILON_Y
    return EPSILON

def AREA_ACO_LONGITUDINAL_CP_RET(M_SD, F_CK, B_W, D, E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a área de aço em elementos de concreto quando submetido a um momento fletor M_SD
    
    TIPO_CONCRETO  | Defina se é concreto protendido ou armado                      |       | string
                   |       'CP' - Concreto protendido                               |       |
                   |       'CA' - Concreto armado                                   |       |
    M_SD           | Momento de cálculo                                             | kN.m  | float
    F_CK           | Resistência característica à compressão                        | kN/m² | float
    B_W            | Largura da viga                                                | m     | float
    D              | Altura útil da seção                                           | m     | float
    E_SCP          | Módulo de elasticidade do aço protendido                       | kN/m² | float
    SIGMA          | Tensão correspondente a tensão EPSILON desejada                | kN/m² | float
    EPSILON_P      | Deformação última do aço                                       |       | float
    EPSILON_Y      | Deformação escoamento do aço                                   |       | float
    F_Y            | Tensão de escoamento do aço                                    | kN/m² | float
    F_P            | Tensão última do aço                                           | kN/m² | float

    Saída:
    X              | Linha neutra da seção medida da parte externa comprimida ao CG | m     | float  
    Z              | Braço de alvanca                                               | m     | float    
    A_S            | Área de aço necessária na seção                                | m²    | float
    EPSILON_S      | Deformação do aço                                              |       | float
    EPSILON_C      | Deformação do concreto                                         |       | float
    """
    # Determinação dos fatores de cálculo de X e A_S
    F_CK /= 1E3
    if F_CK >  50:
        LAMBDA = 0.80 - ((F_CK - 50) / 400)
        ALPHA_C = (1.00 - ((F_CK - 50) / 200)) * 0.85
        EPSILON_C2 = 2.0 + 0.085 * (F_CK - 50) ** 0.53
        EPSILON_C2 = EPSILON_C2 / 1000
        EPSILON_CU = 2.6 + 35.0 * ((90 - F_CK) / 100) ** 4
        EPSILON_CU = EPSILON_CU / 1000
        KX_23 = EPSILON_CU / (EPSILON_CU + 10 / 1000)
        KX_34 = 0.35
    else:
        LAMBDA = 0.80
        ALPHA_C = 0.85
        EPSILON_C2 = 2.0 / 1000
        EPSILON_CU = 3.5 / 1000
        KX_23 = EPSILON_CU / (EPSILON_CU + 10 / 1000)
        KX_34 = 0.45
    # Linhas neutra X
    F_CK *= 1E3
    F_CD = F_CK / 1.40
    PARTE_1 = M_SD / (B_W * ALPHA_C * F_CD)
    NUMERADOR = D - np.sqrt(D ** 2 - 2 * PARTE_1)
    DENOMINADOR = LAMBDA
    X = NUMERADOR / DENOMINADOR
    # Deformações nas fibras comprimidas (concreto) e tracionadas (aço) 
    KX = X / D
    if KX > KX_23:
        EPSILON_C = EPSILON_CU
        EPSILON_S = (1 -  KX) * EPSILON_C 
    elif KX < KX_23:
        EPSILON_S = 10 / 1000
        EPSILON_C = EPSILON_S / (1 - KX)
    elif KX == KX_23:
        EPSILON_S = 10 / 1000
        EPSILON_C = EPSILON_CU 
    # Braço de alavanca Z
    Z = D - 0.50 * LAMBDA * X
    # Área de aço As
    EPSILON_SAUX = DEFORMACAO_ACO(E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y)
    EPSILON_ST = EPSILON_S + EPSILON_SAUX
    F_YD = TENSAO_ACO(E_SCP, EPSILON_ST, EPSILON_P, EPSILON_Y, F_P, F_Y)
    A_S = M_SD / (Z * F_YD)
    return X, EPSILON_S, EPSILON_C, Z, A_S

def AREA_ACO_LONGITUDINAL_CP_T(M_SD, F_CK, B_W, B_F, H_F, D, E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a área de aço em elementos de concreto quando submetido a um momento fletor M_SD
    
    TIPO_CONCRETO  | Defina se é concreto protendido ou armado                      |       | string
                   |       'CP' - Concreto protendido                               |       |
                   |       'CA' - Concreto armado                                   |       |
    M_SD           | Momento de cálculo                                             | kN.m  | float
    F_CK           | Resistência característica à compressão                        | kN/m² | float
    B_W            | Largura da viga                                                | m     | float
    B_F            | Largura da mesa                                                | m     | float
    H_F            | Altura da mesa                                                 | m     | float
    D              | Altura útil da seção                                           | m     | float
    E_SCP          | Módulo de elasticidade do aço protendido                       | kN/m² | float
    SIGMA          | Tensão correspondente a tensão EPSILON desejada                | kN/m² | float
    EPSILON_P      | Deformação última do aço                                       |       | float
    EPSILON_Y      | Deformação escoamento do aço                                   |       | float
    F_Y            | Tensão de escoamento do aço                                    | kN/m² | float
    F_P            | Tensão última do aço                                           | kN/m² | float

    Saída:
    X              | Linha neutra da seção medida da parte externa comprimida ao CG | m     | float  
    Z              | Braço de alvanca                                               | m     | float    
    A_S            | Área de aço necessária na seção                                | m²    | float
    EPSILON_S      | Deformação do aço                                              |       | float
    EPSILON_C      | Deformação do concreto                                         |       | float
    """
    # Determinação dos fatores de cálculo de X e A_S
    F_CK /= 1E3
    if F_CK >  50:
        LAMBDA = 0.80 - ((F_CK - 50) / 400)
        ALPHA_C = (1.00 - ((F_CK - 50) / 200)) * 0.85
        EPSILON_C2 = 2.0 + 0.085 * (F_CK - 50) ** 0.53
        EPSILON_C2 = EPSILON_C2 / 1000
        EPSILON_CU = 2.6 + 35.0 * ((90 - F_CK) / 100) ** 4
        EPSILON_CU = EPSILON_CU / 1000
        KX_23 = EPSILON_CU / (EPSILON_CU + 10 / 1000)
        KX_34 = 0.35
    else:
        LAMBDA = 0.80
        ALPHA_C = 0.85
        EPSILON_C2 = 2.0 / 1000
        EPSILON_CU = 3.5 / 1000
        KX_23 = EPSILON_CU / (EPSILON_CU + 10 / 1000)
        KX_34 = 0.45
    # Linhas neutra X
    F_CK *= 1E3
    F_CD = F_CK / 1.40
    B_WTESTE = B_F
    PARTE_1 = M_SD / (B_WTESTE * ALPHA_C * F_CD)
    NUMERADOR = D - np.sqrt(D ** 2 - 2 * PARTE_1)
    DENOMINADOR = LAMBDA
    X = NUMERADOR / DENOMINADOR
    if (LAMBDA * X) <= H_F:
        # Deformações nas fibras comprimidas (concreto) e tracionadas (aço) 
        KX = X / D
        if KX > KX_23:
            EPSILON_C = EPSILON_CU
            EPSILON_S = (1 -  KX) * EPSILON_C 
        elif KX < KX_23:
            EPSILON_S = 10 / 1000
            EPSILON_C = EPSILON_S / (1 - KX)
        elif KX == KX_23:
            EPSILON_S = 10 / 1000
            EPSILON_C = EPSILON_CU 
        # Braço de alavanca Z
        Z = D - 0.50 * LAMBDA * X
        # Área de aço As
        EPSILON_SAUX = DEFORMACAO_ACO(E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y)
        EPSILON_ST = EPSILON_S + EPSILON_SAUX
        F_YD = TENSAO_ACO(E_SCP, EPSILON_ST, EPSILON_P, EPSILON_Y, F_P, F_Y)
        A_S = M_SD / (Z * F_YD)
    elif (LAMBDA * X) > H_F:
        # Deformações nas fibras comprimidas (concreto) e tracionadas (aço) 
        KX = X / D
        if KX > KX_23:
            EPSILON_C = EPSILON_CU
            EPSILON_S = (1 -  KX) * EPSILON_C 
        elif KX < KX_23:
            EPSILON_S = 10 / 1000
            EPSILON_C = EPSILON_S / (1 - KX)
        elif KX == KX_23:
            EPSILON_S = 10 / 1000
            EPSILON_C = EPSILON_CU 
        # Braço de alavanca Z
        Z = D - 0.50 * LAMBDA * X
        # Área de aço As
        EPSILON_SAUX = DEFORMACAO_ACO(E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y)
        EPSILON_ST = EPSILON_S + EPSILON_SAUX
        F_YD = TENSAO_ACO(E_SCP, EPSILON_ST, EPSILON_P, EPSILON_Y, F_P, F_Y)
        M_1SD = (B_F - B_W) * H_F * ALPHA_C * F_CD * (D - 0.50 * H_F)
        M_2SD = M_SD - M_1SD
        A_1S = M_1SD / ((D - 0.50 * H_F) * F_YD)
        A_2S = M_2SD / (Z * F_YD)
        A_S = A_1S + A_2S
    return X, EPSILON_S, EPSILON_C, Z, A_S

def VERIFICA_ARMADURA_FLEXAO(A_SCP, A_SCPNEC):
    """
    Esta função verifica a restrição do esforço na biela de compressão.
    
    Entrada:
    A_SCP      | Armadura de protensão da peça                      | m²    | float
    A_SCPNEC   | Armadura de protensão necessária para peça         | m²    | float

    Saída:
    G_0        | Valor da restrição analisando a armadura de flexão |       | float
    """
    G_0 = (A_SCPNEC / A_SCP) - 1 
    return G_0   

def MOMENTO_MINIMO(W_INF, F_CTKSUPJ):
    """
    Esta função calcula o momento mínimo para gerar a área de aço mínima.

    Entrada:
    W_INF      | Modulo de resistência inferior                         | m³     | float
    F_CTKSUPJ  | Resistência média caracteristica a tração sup idade J  | kN/m²  | float

    Saída:
    M_MIN      | Momento mínimo para armadura mínima                    | kN.m  | float
    """
    M_MIN = 0.80 * W_INF * F_CTKSUPJ
    return M_MIN

def ARMADURA_ASCP_ELS(A_C, I_C, Y_I, E_P, PSI1_Q1, PSI2_Q1, M_G1, M_G2, M_G3, M_Q1, SIGMA_PI, F_CTKINFJ, FATOR_SEC):
    """
    Esta função calcula a área de aço mínima em função dos limites do ELS.

    Entrada:
    A_C         | Área da  seção transversal da viga                       | m²      | float
    I_C         | Inércia da viga                                          | m^4     | float
    Y_I         | Distância do CG que deseja-se calcular a tensão          | m       | float
    E_P         | Excentricidade de protensão                              | m       | float
    PSI1_Q1     | Coeficiente parcial de segurança PSI_1                   |         | float
    PSI2_Q1     | Coeficiente parcial de segurança PSI_2                   |         | float
    M_          | Momentos caracteristicos da peça (G,Q)                   | kN.m    | float
    SIGMA_PI    | Tensão de protensão                                      | kN/m²   | float
    F_CTKINFJ   | Resistência caracteristica a tração inferior na idade j  | kN/m²   | float
    FATOR_SEC   | Fator de correção da resistência                         |         | float

    Saída:
    A_SCPINICIAL| Área de aço inicial respeitando os limites de serviço    | m²      | float
    """
    # ELS-F
    if FATOR_SEC == 'RETANGULAR':
        ALPHA_F = 1.50
    elif FATOR_SEC == 'I':
        ALPHA_F = 1.30
    elif FATOR_SEC == 'DUPLO T':
        ALPHA_F = 1.20
    LIMITE_TRAC0 = - ALPHA_F * F_CTKINFJ
    AUX_0 = SIGMA_PI / A_C + (SIGMA_PI * E_P * Y_I) / I_C
    AUX_1 = ((M_G1 + M_G2 + M_G3) * Y_I) / I_C + ((PSI1_Q1 * M_Q1) * Y_I) / I_C
    A_SCP0 = (LIMITE_TRAC0 +  AUX_1) / AUX_0
    # ELS-D
    LIMITE_TRAC1 = 0
    AUX_2 = ((M_G1 + M_G2 + M_G3) * Y_I) / I_C + ((PSI2_Q1 * M_Q1) * Y_I) / I_C
    A_SCP1 = (LIMITE_TRAC1 + AUX_2) / AUX_0
    A_SCPINICIAL = max(A_SCP0, A_SCP1)
    return A_SCPINICIAL

def ARMADURA_ASCP_ELU(A_C, W_INF, W_SUP, E_P, PSI1_Q1, PSI2_Q1, M_G1, M_G2, M_G3, M_Q1, SIGMA_PI, F_CTMJ, F_CKJ):
    """
    Esta função calcula a área de aço mínima em função dos limites do ELU.

    Entrada:
    A_C         | Área da  seção transversal da viga                       | m²      | float
    W_INF       | Modulo de resistência inferior                           | m³      | float
    W_SUP       | Modulo de resistência superior                           | m³      | float
    E_P         | Excentricidade de protensão                              | m       | float
    PSI1_Q1     | Coeficiente parcial de segurança PSI_1                   |         | float
    PSI2_Q1     | Coeficiente parcial de segurança PSI_2                   |         | float
    M_          | Momentos caracteristicos da peça (G,Q)                   | kN.m    | float
    SIGMA_PI    | Tensão de protensão                                      | kN/m²   | float
    F_CTMJ      | Resistência caracteristica a tração média na idade j     | kN/m²   | float
    F_CKJ       | Resistência característica à compressão idade j          | kN/m²   | float 

    Saída:
    A_SCPINICIAL| Área de aço inicial respeitando os limites do ato de     | m²      | float
                  protensão  
    """
    LIMITE_TRAC0 = -1.20 * F_CTMJ
    AUX_0 = SIGMA_PI / A_C - (SIGMA_PI * E_P) / W_SUP
    AUX_1 = (M_G1 + M_G2 + M_G3) / W_SUP + (PSI1_Q1 * M_Q1) / W_SUP
    A_SCP0 = (LIMITE_TRAC0 -  AUX_1) / AUX_0
    LIMITE_COMP0 = 0.70 * F_CKJ
    AUX_2 = (M_G1 + M_G2 + M_G3) / W_INF + (PSI2_Q1 * M_Q1) / W_INF
    AUX_3 = SIGMA_PI / A_C + (SIGMA_PI * E_P) / W_INF
    A_SCP1 = (LIMITE_COMP0 + AUX_2) / AUX_3
    return A_SCP0, A_SCP1

"""
def ABERTURA_FISSURAS(ALFA_E, P_IINF, A_2, M_SDMAX, D, X_2, I_2, DIAMETRO_ARMADURA, ETA_COEFICIENTE_ADERENCIA, E_SCP, F_CTM, RHO_R) :
    Esta função calcula a abertura de fissuras na peça 

    Entrada:
    ALFA_E      | Relação dos modulos                                    | m²      | float
    W_INF       | Modulo de resistência inferior                         | m³      | float
    PSI1_Q1     | Coeficiente parcial de segurança PSI_1                 |         | float
    PSI2_Q1     | Coeficiente parcial de segurança PSI_2                 |         | float
  
    SIGMA_S = ALFA_E * (P_IINF / A_2) + ( ALFA_E * (M_SDMAX * (D - X_2) / I_2 ) )
    W_1 = (DIAMETRO_ARMADURA / 12.5 * ETA_COEFICIENTE_ADERENCIA) * (SIGMA_S / E_SCP) * 3 (SIGMA_S / F_CTM)
    W_2 = (DIAMETRO_ARMADURA / 12.5 * ETA_COEFICIENTE_ADERENCIA) * (SIGMA_S / E_SCP) * ((4 / RHO_R) + 45)
    W_FISSURA = min(W_1, W_2)
    return W_FISSURA
"""

def PROP_GEOMETRICA_ESTADIO_I(H, B_F, B_W, H_F, A_SB, ALPHA_MOD, D):
    """
    Esta função calcula as propriedades geométricas no estádio I.

    Entrada:
    H         | Altura da viga                                     | m    | float
    B_F       | Base de mesa superior da seção                     | m    | float
    B_W       | Base de alma da seção                              | m    | float
    H_F       | Altura de mesa superior da viga                    | m    | float
    A_SB      | Area de aço na seção tracionada                    | m²   | float
    ALPHA_MOD | Relação entre os modulos                           |      | float
    D         | Altura útil da seção                               | m    | float

    Saída:
    A_C       | Área de concreto no estádio 1                      | m²   | float
    X_I       | Centro geometrico da viga no estádio 1             | m    | float
    I_I       | Inércia da viga no estádio 1                       | m^4  | float
    """
    A_C = (B_F - B_W) * H_F + B_W * H + A_SB * (ALPHA_MOD - 1)
    X_I = ((B_F - B_W) * ((H_F ** 2) / 2) + B_W * ((H ** 2 ) / 2) + A_SB * (ALPHA_MOD - 1) * D) / A_C
    I_I = ((B_F - B_W) * H_F ** 3) / 12 + (B_W * H ** 3) / 12 + (B_F - B_W) * H_F * (X_I - H_F / 2) ** 2 + B_W * H * (X_I - H / 2) ** 2 + A_SB * (ALPHA_MOD - 1) * (X_I - D) ** 2
    return A_C, X_I, I_I

def PROP_GEOMETRICA_ESTADIO_II(H, B_F, B_W, H_F, A_SB, A_ST, ALPHA_MOD, D, D_L):
    """
    Esta função calcula as propriedades geométricas no estádio 2.

    Entrada:
    H         | Altura da viga                                     | m    | float
    B_F       | Base de mesa superior da seção                     | m    | float
    B_W       | Base de alma da seção                              | m    | float
    H_F       | Altura de mesa superior da viga                    | m    | float
    A_SB      | Area de aço na seção tracionada                    | m²   | float
    A_ST      | Area de aço na seção comprimida                    | m²   | float
    ALPHA_MOD | Relação entre os modulos                           |      | float
    D         | Altura útil da seção                               | m    | float
    D_L       | Altura útil da armadura comprimida                 | m    | float

    Saída:
    X_II      | Centro geometrico da viga no estádio 2             | m    | float
    I_II      | Inércia da viga no estádio 2                       | m^4  | float
    """
    A_1 = B_F / 2
    A_2 = H_F * (0) + (ALPHA_MOD - 1) * A_ST + ALPHA_MOD * A_SB
    A_3 = -D_L * (ALPHA_MOD - 1) * A_ST - D * ALPHA_MOD * A_SB - (H_F ** 2) / 2 * (0)
    X_II = (- A_2 + (A_2 ** 2 - 4 * A_1 * A_3) ** 0.50) / (2 * A_1)
    if X_II <= H_F:
        pass
    elif X_II > H_F:
        A_1 = B_W / 2
        A_2 = H_F * (0) + (ALPHA_MOD - 1) * A_ST + ALPHA_MOD * A_SB
        A_3 = -D_L * (ALPHA_MOD - 1) * A_ST - D * ALPHA_MOD * A_SB - (H_F ** 2) / 2 * (0)
        X_II = (- A_2 + (A_2 ** 2 - 4 * A_1 * A_3) ** 0.50) / (2 * A_1)
    if X_II <= H_F:
        I_II = (B_F * X_II ** 3) / 3 + ALPHA_MOD * A_SB * (X_II - D) ** 2 + (ALPHA_MOD - 1) * A_ST * (X_II - D_L) ** 2
    else:
        I_II = ((B_F - B_W) * H_F ** 3) / 12 + (B_W * X_II **3 ) / 3 + (B_F - B_W) * (X_II - H_F / 2) ** 2 + ALPHA_MOD * A_SB * (X_II - D) ** 2 + (ALPHA_MOD - 1) * A_ST * (X_II - D_L) ** 2
    return X_II, I_II

def INERCIA_BRANSON(M_R, M_D, I_I, I_II):
    """
    Esta função calcula o momento de Branson.

    Entrada:
    M_R       | Momento resistente                                 | kN.m | float
    M_D       | Momento de cálculo                                 | kN.m | float
    I_I       | Inércia da viga no estádio 1                       | m^4  | float
    I_II      | Inércia da viga no estádio 2                       | m^4  | float

    Saída:
    I_BRANSON | Inércia de Branson                                 | m^4  | float
    """
    M_RMD = (M_R / M_D) ** 3
    I_BRANSON = M_RMD * I_I + (1 - M_RMD) * I_II
    return I_BRANSON

def FLECHA_DIRETA(TIPO_VIGA, EI, P_K, L):
    """
    Esta função calcula a fleha direta.

    Entrada:
    TIPO_VIGA | Tipo de vinculação da viga                         |      | float
    EI        | Rigidez da flexão                                  | kN/m²| float
    P_K       | Carga linearmente distribuida                      | kN   | float
    L         | Comprimento do vão                                 | m    | float

    Saída:
    FLECHA    | Valor da flecha                                    | m    | float
    """
    if TIPO_VIGA == 0:
        FLECHA = 5 * P_K * (L ** 4) * (1 / (384 * EI))
    elif TIPO_VIGA == 1:
        FLECHA = 1 * P_K * (L ** 3) * (1 / (48 * EI))
    return FLECHA


def MOMENTO_RESISTENTE(FATOR_SEC, F_CT, H, X_I, I_I, P_I, A_C, W_INF, E_P):
    """
    Esta função determina o momento resistente
   
    Entrada:
    FATOR_SEC      | Define o modelo de viga empregado                              |       | string
                   |       'RETANGULAR' - Viga no formato retangular                |       |      
                   |       'I' - Viga no formato I                                  |       |
                   |       'DUPLO T' - Viga no formato duplo t                      |       |
    F_CT           | Resistência caracteristica a tração                            | kN/m² | float
    H              | Altura da viga                                                 | m     | float
    X_I            | Centro geometrico da viga no estádio 1                         | m     | float
    I_I            | Inércia da viga no estádio 1                                   | m^4   | float
    P_I            | Carga de protensão considerando as perdas                      | kN    | float
    A_C            | Área da seção transversal da viga                              | m²    | float
    W_INF          | Modulo de resistência inferior                                 | m³    | float
    E_P            | Excentricidade de protensão                                    | m²    | float

    Saída:
    M_R            | Valor do momento resistente                                    | kN.m  | float
    """                   
    if FATOR_SEC == 'RETANGULAR':
        ALPHA_F = 1.50
    elif FATOR_SEC == 'I':
        ALPHA_F = 1.30
    elif FATOR_SEC == 'DUPLO T':
        ALPHA_F = 1.20
    Y_T = H - X_I
    AUX = (1 / A_C) + (E_P / W_INF)
    M_0 = P_I * W_INF * AUX
    M_R = M_0 + ALPHA_F * F_CT * (I_I / Y_T)
    return M_R

def VERIFICA_FLECHA(A_TOTAL, A_MAX):
    """
    Esta função realiza a verificação da viga para flecha.

    Entrada:
    A_TOTAL        | Valor da flecha total                              | m    | float
    A_MAX          | Valor da flecha máxima                             | m    | float
    Saída:
    G_0            | Valor da restrição analisando a flecha             |      | float
    """
    G_0 = (A_TOTAL / A_MAX) - 1 
    return G_0

def PERDA_DESLIZAMENTO_ANCORAGEM(P_IT0, SIGMA_PIT0, A_SCP, L_0, DELTA_ANC, E_SCP):
    """
    Esta função determina a perda de protensão por deslizamento da armadura na anco-
    ragem.
    
    Entrada:
    P_IT0       | Carga inicial de protensão                        | kN    | float
    SIGMA_PIT0  | Tensão inicial de protensão                       | kN/m² | float
    A_SCP       | Área de total de armadura protendida              | m²    | float
    L_0         | Comprimento da pista de protensão                 | m     | float
    DELTA_ANC   | Previsão do deslizamento do sistema de ancoragem  | m     | float
    E_SCP       | Módulo de Young do aço protendido                 | kN/m² | float

    Saída:
    DELTAPERC   | Perda percentual de protensão                     | %     | float
    P_IT1       | Carga final de protensão                          | kN    | float
    SIGMA_PIT1  | Tensão inicial de protensão                       | kN/m² | float
    """
    # Pré-alongamento do cabo
    DELTAL_P = L_0 * (SIGMA_PIT0 / E_SCP)
    # Redução da deformação na armadura de protensão
    DELTAEPSILON_P = DELTA_ANC / (L_0 +  DELTAL_P)
    # Perdas de protensão
    DELTASIGMA = E_SCP * DELTAEPSILON_P
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def PERDA_DEFORMACAO_CONCRETO(E_SCP, E_CCP, P_IT0, SIGMA_PIT0, A_SCP, A_C, I_C, E_P, M_GPP):
    """
    Esta função determina a perda de protensão devido a deformação inicial do concreto. 
    
    Entrada:
    E_SCP       | Módulo de Young do aço protendido                 | kN/m² | float
    E_CCP       | Módulo de Young do concreto                       | kN/m² | float
    P_IT0       | Carga inicial de protensão                        | kN    | float
    SIGMA_PIT0  | Tensão inicial de protensão                       | kN/m² | float
    A_C         | Área bruta da seção                               | m²    | float
    A_SCP       | Área de total de armadura protendida              | m²    | float
    I_C         | Inércia da seção bruta                            | m^4   | float
    E_P         | Excentricidade de protensão                       | m     | float 
    M_GPP       | Momento fletor devido ao peso próprio             | kN.m  | float 
      
    Saída:
    DELTAPERC   | Perda percentual de protensão                     | %     | float
    P_IT1       | Carga final de protensão                          | kN    | float
    SIGMA_PIT1  | Tensão inicial de protensão                       | kN/m² | float
    """
    # Perdas de protensão
    ALPHA_P = E_SCP / E_CCP
    AUX_0 = P_IT0 / A_C
    AUX_1 = (P_IT0 * E_P ** 2) / I_C
    AUX_2 = (M_GPP * E_P) / I_C
    DELTASIGMA = ALPHA_P * (AUX_0 + AUX_1 - AUX_2)
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def INTERPOLADOR(X_1, X_2, X_K, Y_1, Y_2):
    """
    Esta função interpola linearmente valores.

    Entrada:
    X_1   | Valor inferior X_K     |       | float
    X_2   | Valor superior X_K     |       | float
    Y_1   | Valor inferior Y_K     |       | float
    Y_2   | Valor superior Y_K     |       | float
    X_K   | Valor X de referência  |       | float

    Saída:
    Y_K   | Valor interpolado Y    |       | float
    """
    Y_K = Y_1 + (X_K - X_1) * ((Y_2 - Y_1) / (X_2 - X_1))
    return Y_K 

def TABELA_PSI1000(TIPO_FIO_CORD_BAR, TIPO_ACO, RHO_SIGMA):
    """
    Esta função encontra o fator Ψ_1000 para cálculo da relaxação.

    Entrada:
    TIPO_FIO_CORD_BAR  | Tipo de armadura de protensão de acordo com a aderência escolhida                 |       | string
                       |    'FIO' - Fio                                                                    |       |
                       |    'COR' - Cordoalha                                                              |       |
                       |    'BAR' - BARRA                                                                  |       |
    TIPO_ACO           | Tipo de aço                                                                       |       | string
                       |    'RN' - Relaxação normal                                                        |       |
                       |    'RB' - Relaxação baixa                                                         |       |
    RHO_SIGMA          | Razão entre F_PK e SIGMA_PI                                                       |       | float

    Saída:
    PSI_1000           | Valor médio da relaxação, medidos após 1.000 h, à temperatura constante de 20 °C  | %     | float     
    """
    # Cordoalhas
    if TIPO_FIO_CORD_BAR == 'COR':
        if TIPO_ACO == 'RN':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 3.50
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 3.50; Y_1 = 7.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 7.00; Y_1 = 12.00
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA 
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)   
        elif TIPO_ACO == 'RB':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 1.30
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 1.30; Y_1 = 2.50
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 2.50; Y_1 = 3.50
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
    # Fio
    elif TIPO_FIO_CORD_BAR == 'FIO':
        if TIPO_ACO == 'RN':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 2.50
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 2.50; Y_1 = 5.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 5.00; Y_1 = 8.50
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA   
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1) 
        elif TIPO_ACO == 'RB':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0 
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 1.00
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 1.00; Y_1 = 2.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 2.00; Y_1 = 3.00
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA  
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)  
    # Barra
    elif TIPO_FIO_CORD_BAR == 'BAR':
        if RHO_SIGMA <= 0.5:
                PSI_1000 = 0 
        elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 1.50
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1) 
        elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 1.50; Y_1 = 4.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1) 
        elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 4.00; Y_1 = 7.00
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA 
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)        
    return PSI_1000  

def PERDA_RELAXACAO_ARMADURA(P_IT0, SIGMA_PIT0, T_0, T_1, TEMP, F_PK, A_SCP, TIPO_FIO_CORD_BAR, TIPO_ACO):
    """
    Esta função determina a perda de protensão por relaxação da armadura de protensão em peças de concreto 
    protendido.
    
    Entrada:
    P_IT0              | Carga inicial de protensão                                         | kN    | float
    SIGMA_PIT0         | Tensão inicial de protensão                                        | kN/m² | float
    T_0                | Tempo inicial de análise sem correção da temperatura               | dias  | float
    T_1                | Tempo final de análise sem correção da temperatura                 | dias  | float 
    TEMP               | Temperatura de projeto                                             | °C    | float 
    F_PK               | Tensão última do aço                                               | kN/m² | float
    A_SCP              | Área de total de armadura protendida                               | m²    | float
    TIPO_FIO_CORD_BAR  | Tipo de armadura de protensão de acordo com a aderência escolhida  |       | string
                       |    'FIO' - Fio                                                     |       |
                       |    'COR' - Cordoalha                                               |       |
                       |    'BAR' - BARRA                                                   |       |
    TIPO_ACO           | Tipo de aço                                                        |       | string
                       |    'RN' - Relaxação normal                                         |       |
                       |    'RB' - Relaxação baixa                                          |       |
      
    Saída:
    DELTAPERC          | Perda percentual de protensão                                      | %     | float
    P_IT1              | Carga final de protensão                                           | kN    | float
    SIGMA_PIT1         | Tensão inicial de protensão                                        | kN/m² | float
    PSI                | Coeficiente de relaxação do aço                                    | %     | float
    """
    # Determinação PSI_1000
    RHO_SIGMA = SIGMA_PIT0 / F_PK
    PSI_1000 = TABELA_PSI1000(TIPO_FIO_CORD_BAR, TIPO_ACO, RHO_SIGMA)     
    # Determinação do PSI no intervalo de tempo T_1 - T_0
    DELTAT_COR = (T_1 - T_0) * TEMP / 20
    if T_1 > (49 * 365):
        PSI =  2.50 * PSI_1000
    else:
        PSI =  PSI_1000 * (DELTAT_COR / 41.67) ** 0.15
    # Perdas de protensão
    DELTASIGMA = (PSI / 100) * SIGMA_PIT0
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1, PSI

def CALCULO_HFIC(U, A_C, MU_AR):
    """
    Esta função calcula a altura fictícia de uma peça de concreto.

    Entrada:
    U       | Umidade do ambiente no intervalo de tempo de análise         | %     | float
    A_C     | Área bruta da seção                                          | m²    | float
    MU_AR   | Parte do perímetro externo da seção em contato com ar        | m     | float

    Saída:
    H_FIC   | Altura fictícia da peça para cálculo de fluência e retração  | m     | float
    """
    GAMMA = 1 + np.exp(-7.8 + 0.1 * U)
    H_FIC = GAMMA * 2 * A_C / MU_AR
    if H_FIC > 1.60:
        H_FIC = 1.60
    if H_FIC < 0.050:
        H_FIC = 0.050
    return H_FIC

def BETAS_RETRACAO(T_FIC, H_FIC):
    """
    Esta função determina o coeficiente de retração β_S.

    Entrada:
    T_FIC                  | Tempo de projeto corrigido em função da temperatura          | dias  | float 
    H_FIC                  | Altura fictícia da peça para cálculo de fluência e retração  | m     | float

    Saída:
    BETA_S, A, B, C , D, E | Coeficientes da retração                                     |       | float
    """
    # Coeficientes A até E
    A = 40
    B = 116 * pow(H_FIC, 3) - 282 * pow(H_FIC, 2) + 220 * H_FIC - 4.8
    C = 2.5 * pow(H_FIC, 3) - 8.8 * H_FIC + 40.7
    D = -75 * pow(H_FIC, 3) + 585 * pow(H_FIC, 2) + 496 * H_FIC - 6.8
    E = -169 * pow(H_FIC, 4) + 88 * pow(H_FIC, 3) + 584 * pow(H_FIC, 2) - 39 * H_FIC + 0.8
    T_FIC100 = T_FIC / 100;
    # Determinação do BETA_S
    AUX_0 = pow(T_FIC100, 3) + A * pow(T_FIC100, 2) + B * T_FIC100
    AUX_1 = pow(T_FIC100, 3) + C * pow(T_FIC100, 2) + D * T_FIC100 + E
    BETA_S =  AUX_0 / AUX_1
    return BETA_S, A, B, C, D, E

def BETAF_FLUENCIA(T_FIC, H_FIC):
    """
    Esta função determina o coeficiente de retração β_F.

    Entrada:
    T_FIC                | Tempo de projeto corrigido em função da temperatura          | dias  | float 
    H_FIC                | Altura fictícia da peça para cálculo de fluência e retração  | m     | float

    Saída:
    BETA_F, A, B, C , D  | Coeficientes da fluência                                     |       | float
    """
    A = 42 * pow(H_FIC, 3) - 350 * pow(H_FIC, 2) + 588 * H_FIC + 113
    B = 768 * pow(H_FIC,3) - 3060 * pow(H_FIC, 2) + 3234 * H_FIC - 23
    C = -200 * pow(H_FIC, 3) + 13 * pow(H_FIC, 2) + 1090 * H_FIC + 183
    D = 7579 * pow(H_FIC,3) - 31916 * pow(H_FIC, 2) + 35343 * H_FIC + 1931
    # Determinação do BETA_F
    AUX_0 = pow(T_FIC, 2) + A * T_FIC + B 
    AUX_1 = pow(T_FIC, 2) + C * T_FIC + D
    BETA_F =  AUX_0 / AUX_1
    return BETA_F, A, B, C , D

def CALCULO_TEMPO_FICTICIO(T, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO):
    """
    Esta função calcula o tempo corrigido para cálculo das perdas de fluência e retração. 

    Entrada:
    T                   | Tempo para análise da correção em função da temperatura    | dias  | float
    TEMP                | Temperatura de projeto                                     | °C    | float 
    TIPO_PERDA          | Tipo da perda que deseja-se calcular a correção do tempo   |       | string
                        |       'LENTO'  - Endurecimento lento AF250, AF320, POZ250  |       |
                        |       'NORMAL' - Endurecimento normal CP250, CP320, CP400  |       |
                        |       'RAPIDO' - Endurecimento rápido aderência            |       |
    TIPO_ENDURECIMENTO  | Tipo de enduricmento do cimento                            |       | string
                        |       'RETRACAO' - Retração                                |       |
                        |       'FLUENCIA' - Fluência                                |       |                                                                                           

    Saída:
    T_FIC               | Tempo de projeto corrigido em função da temperatura        | °C    | float 
    """
    # Parâmetros de reologia e tipo de pega
    if TIPO_PERDA == 'RETRACAO':
        ALFA = 1
    elif TIPO_PERDA == 'FLUENCIA':
        if TIPO_ENDURECIMENTO == 'LENTO':
            ALFA = 1
        elif TIPO_ENDURECIMENTO == 'NORMAL':
            ALFA = 2
        elif TIPO_ENDURECIMENTO == 'RAPIDO':
            ALFA = 3
    # Correção dos tempos menores que 3 dias e maiores que 10.000 dias
    if T < 3 and T > 0:
        T = 3
    elif T > 10000:
        T = 10000
    # Determinação da idade fictícia do concreto
    T_FIC = ALFA * ((TEMP + 10) / 30) * T
    return T_FIC 

def PERDA_RETRACAO_CONCRETO(P_IT0, SIGMA_PIT0, A_SCP, U, ABAT, A_C, MU_AR, T_0, T_1, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO, E_SCP):
    """
    Esta função determina a perda de protensão devido a retração do concreto. 
    
    Entrada:
    P_IT0               | Carga inicial de protensão                                 | kN    | float
    SIGMA_PIT0          | Tensão inicial de protensão                                | kN/m² | float
    A_SCP               | Área de total de armadura protendida                       | m²    | float
    U                   | Umidade do ambiente no intervalo de tempo de análise       | %     | float
    ABAT                | Abatimento ou slump test do concreto                       | kN/m² | float
    A_C                 | Área bruta da seção                                        | m²    | float 
    MU_AR               | Parte do perímetro externo da seção em contato com ar      | m     | float
    T_0                 | Tempo inicial de análise sem correção da temperatura       | dias  | float
    T_1                 | Tempo final de análise sem correção da temperatura         | dias  | float 
    TEMP                | Temperatura de projeto                                     | °C    | float 
    TIPO_PERDA          | Tipo da perda que deseja-se calcular a correção do tempo   |       | string
                        |       'LENTO'  - Endurecimento lento AF250, AF320, POZ250  |       |
                        |       'NORMAL' - Endurecimento normal CP250, CP320, CP400  |       |
                        |       'RAPIDO' - Endurecimento rápido aderência            |       |
    TIPO_ENDURECIMENTO  | Tipo de enduricmento do cimento                            |       | string
                        |       'RETRACAO' - Retração                                |       |
                        |       'FLUENCIA' - Fluência                                |       |   
    E_SCP               | Módulo de Young do aço protendido                          | kN/m² | float                                                                                        
      
    Saída:
    DELTAPERC           | Perda percentual de protensão                              | %     | float
    P_IT1               | Carga final de protensão                                   | kN    | float
    SIGMA_PIT1          | Tensão inicial de protensão                                | kN/m² | float
    """
    # Cálculo da defomração específica EPSILON_1S
    EPSILON_1S = -8.09 + (U / 15) - (U ** 2 / 2284) - (U ** 3 / 133765) + (U ** 4 / 7608150)
    EPSILON_1S /= -1E4 
    if U <= 90 and (ABAT >= 0.05 and ABAT <= 0.09):          # intervalo 0.05 <= ABAT <= 0.09
        EPSILON_1S *= 1.00
    elif U <= 90 and (ABAT >= 0.00 and ABAT <= 0.04):        # intervalo 0.00 <= ABAT <= 0.04
        EPSILON_1S *= 0.75
    elif U <= 90 and (ABAT >= 0.10 and ABAT <= 0.15):        # intervalo 10.0 <= ABAT <= 15.0
        EPSILON_1S *= 1.25
    # Cálculo da defomração específica EPSILON_2S
    H_FIC = CALCULO_HFIC(U, A_C, MU_AR)
    EPSILON_2S = (0.33 + 2 * H_FIC) / (0.208 + 3 * H_FIC)
    # Coeficiente BETA_S T0
    T_0FIC = CALCULO_TEMPO_FICTICIO(T_0, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO)
    BETA_S0, _, _, _, _, _, = BETAS_RETRACAO(T_0FIC, H_FIC)
    # Coeficiente BETA_S T1
    T_1FIC = CALCULO_TEMPO_FICTICIO(T_1, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO)
    BETA_S1, _, _, _, _, _, = BETAS_RETRACAO(T_1FIC, H_FIC)
    # Valor final da deformação por retração EPSILON_CS
    EPSILON_CS = EPSILON_1S * EPSILON_2S * (BETA_S1 - BETA_S0)
    # Perdas de protensão
    DELTASIGMA = E_SCP * EPSILON_CS
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def PHI_FLUENCIA(F_CKJ, F_CK, U, A_C, MU_AR, ABAT, T_0, T_1, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO):
    """
    Esta função determina os fatores de fluência φ(t, t_0) de estruturas de concreto.

    Entrada:
    F_CK                | Resistência característica à compressão                    | kN/m² | float   
    F_CKJ               | Resistência característica à compressão idade J            | kN/m² | float
    U                   | Umidade do ambiente no intervalo de tempo de análise       | %     | float
    A_C                 | Área bruta da seção                                        | m²    | float 
    MU_AR               | Parte do perímetro externo da seção em contato com ar      | m     | float
    ABAT                | Abatimento ou slump test do concreto                       | kN/m² | float
    T_0                 | Tempo inicial de análise sem correção da temperatura       | dias  | float
    T_1                 | Tempo final de análise sem correção da temperatura         | dias  | float 
    TEMP                | Temperatura de projeto                                     | °C    | float 
    TIPO_PERDA          | Tipo da perda que deseja-se calcular a correção do tempo   |       | string
                        |       'LENTO'  - Endurecimento lento AF250, AF320, POZ250  |       |
                        |       'NORMAL' - Endurecimento normal CP250, CP320, CP400  |       |
                        |       'RAPIDO' - Endurecimento rápido aderência            |       |
    TIPO_ENDURECIMENTO  | Tipo de enduricmento do cimento                            |       | string
                        |       'RETRACAO' - Retração                                |       |
                        |       'FLUENCIA' - Fluência                                |       |   
    
    Saída:
    PHI                 | Fator de fluência                                          |       | float
    """
    # Fator PHI_A
    F_CK /= 1E3
    if F_CK <= 45:
        F_CK *= 1E3
        PHI_A = 0.80 * (1 - F_CKJ / F_CK)
    elif F_CK > 45: 
        F_CK *= 1E3
        PHI_A = 1.40 * (1 - F_CKJ / F_CK)
    # Fator PHI_1C
    PHI_1C = 4.45 - 0.035 * U
    if U <= 90 and (ABAT >= 0.05 and ABAT <= 0.09):        # intervalo 0.05 <= ABAT <= 0.09
        PHI_1C *= 1.00
    elif U <= 90 and (ABAT >= 0.00 and ABAT <= 0.04):      # intervalo 0.00 <= ABAT <= 0.04
        PHI_1C *= 0.75
    elif U <= 90 and (ABAT >= 10.0 and ABAT <= 15.0):      # intervalo 10.0 <= ABAT <= 15.0
        PHI_1C *= 1.25
    # Fator PHI_2C
    H_FIC = CALCULO_HFIC(U, A_C, MU_AR) 
    PHI_2C = (0.42 + H_FIC) / (0.20 + H_FIC)
    F_CK /= 1E3
    if F_CK <= 45:
        PHI_F = PHI_1C * PHI_2C
    elif F_CK > 45:
        PHI_F = 0.45 * PHI_1C * PHI_2C
    F_CK *= 1E3
    # Fator PHI_D
    PHI_D = 0.40
    # Cálculo fatores BETA
    T_0FIC = CALCULO_TEMPO_FICTICIO(T_0, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO)
    T_1FIC = CALCULO_TEMPO_FICTICIO(T_1, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO)
    DELTA_T = T_1FIC - T_0FIC
    BETA_D = (DELTA_T + 20) / (DELTA_T + 70)
    BETA_F0, _, _, _, _ = BETAF_FLUENCIA(T_0FIC, H_FIC)
    BETA_F1, _, _, _, _ = BETAF_FLUENCIA(T_1FIC, H_FIC)
    # Coeficiente de fluência
    PHI = PHI_A + PHI_F * (BETA_F1 - BETA_F0) + PHI_D * BETA_D
    return PHI

def PERDA_POR_FLUENCIA_NO_CONCRETO(P_IT0, SIGMA_PIT0, A_SCP, PHI, E_SCP, E_CCP28, SIGMA_CABO):
    """
    Esta função determina a perda de protensão devido ao efeito de fluência.

    Entrada:
    PHI         | Fator de fluência para cada carregamento estudado  |       | Py list
    E_SCP       | Módulo de Young do aço protendido                  | kN/m² | float
    E_CCP28     | Módulo de Young do concreto aos 28 dias            | kN/m² | float
    SIGMA_CABO  | Tensões no nível dos cabos                         | kN/m² | Py list
    P_IT0       | Carga inicial de protensão                         | kN    | float
    SIGMA_PIT0  | Tensão inicial de protensão                        | kN/m² | float
    A_SCP       | Área de total de armadura protendida               | m²    | float

    Saída:
    DELTAPERC   | Perda percentual de protensão                      | %     | float
    P_IT1       | Carga final de protensão                           | kN    | float
    SIGMA_PIT1  | Tensão inicial de protensão                        | kN/m² | float
    """
    TAM = len(PHI)
    # Perdas de protensão
    AUX = 0
    for I_CONT in range(TAM):
        AUX += SIGMA_CABO[I_CONT] * PHI[I_CONT]
    ALPHA_P = E_SCP / E_CCP28
    DELTASIGMA = ALPHA_P * AUX
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def INTERACAO_ENTRE_PERDAS_PROGRESSIVAS(E_P, A_C, I_C, E_SCP, E_CCP28, DELTASIGMA_RETRACAO, DELTASIGMA_FLUENCIA, P_IT0, SIGMA_PIT0, PSI, A_SCP, PHI_0):
    """
    Esta função determina a perdas progressivas de protensão considerando o efeito conjunto
    das perdas.

    Entrada:
    A_C                 | Área da  seção transversal da viga         | m²    | float
    I_C                 | Inércia da viga                            | m^4   | float
    E_P                 | Excentricidade de protensão                | m     | float 
    E_SCP               | Módulo de Young do aço protendido          | kN/m² | float
    E_CCP28             | Módulo de Young do concreto idade 28 dias  | kN/m² | float
    DELTASIGMA_RETRACAO | Perda de protensão devido a retração       | kN/m² | float
    DELTASIGMA_FLUENCIA | Perda de protensão devido a fluência       | kN/m² | float
    P_IT0               | Carga inicial de protensão                 | kN    | float
    SIGMA_PIT0          | Tensão inicial de protensão                | kN/m² | float
    PSI                 | Coeficiente de relaxação do aço            | %     | float    
    A_SCP               | Área de total de armadura protendida       | m²    | float
    PHI_0               | Fator de fluência carga P_i e Permanente   |       | float
                        |   no instante t_0                          |       | 
    Saída:
    DELTAPERC           | Perda percentual de protensão              | %     | float
    P_IT1               | Carga final de protensão                   | kN    | float
    SIGMA_PIT1          | Tensão inicial de protensão                | kN/m² | float
    """
    # Perdas de protensão
    PSI /= 1E2
    CHI = - np.log(1 - PSI)
    CHI_P = 1 + CHI
    CHI_C = 1 + 0.50 * PHI_0
    ALPHA_P = E_SCP / E_CCP28
    ETA = 1 + ((E_P ** 2) * (A_C / I_C))
    PHO_P = A_SCP / A_C
    AUX_0 =  DELTASIGMA_RETRACAO + DELTASIGMA_FLUENCIA + SIGMA_PIT0 * CHI
    AUX_1 = CHI_P + CHI_C * ALPHA_P * ETA * PHO_P
    DELTASIGMA = AUX_0 / AUX_1
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def VERIFICACAO_VIGA(VIGA):
    """
    Esta função verifica as a quantidade de armadura de uma viga pré-fabricada e protendida. Além verifica se essa viga atende os requisitos de Estado Limite.
    
    Entrada:
    VIGA      |  Dados da viga modelo                                 |      | Py dictionary
              |    Etiquetas do dicionário                            |      | 
              |    'H_W' : Altura da alma                             | m    | Float
              |    'B_FS': Base de mesa superior da viga              | m    | Float
              |    'B_FI': Base de mesa inferior da viga              | m    | float
              |    'B_W' : Base de alma da viga                       | m    | float
              |    'H_S' : Altura de mesa superior da viga            | m    | float
              |    'H_I' : Altura de mesa inferior da viga            | m    | float
              |    'H_SI': Altura inclinada de mesa superior da viga  | m    | float
              |    'H_II': Altura inclinada de mesa inferior da viga  | m    | float
    """
    IMPRESSAO = VIGA['IMPRESSÃO']
    H_W = VIGA['H_W']
    B_W = VIGA['B_W']
    B_FS = VIGA['B_FS']
    B_FI = VIGA['B_FI']
    H_FS = VIGA['H_FS']
    H_FI = VIGA['H_FI']
    H_SI = VIGA['H_SI']
    H_II = VIGA['H_II']
    COB = VIGA['COB']
    PHI_L = VIGA['PHI_L']
    PHI_E = VIGA['PHI_E']
    L = VIGA['L']
    L_PISTA = VIGA['L_PISTA']
    FATOR_SEC = VIGA['FATOR_SEC']
    DELTA_ANC = VIGA['DELTA_ANC']
    TEMPO_CONC = VIGA['TEMPO_CONC']
    TEMPO_ACO = VIGA['TEMPO_ACO']
    TEMP = VIGA['TEMP']
    PERDA_INICIAL = VIGA['PERDA_INICIAL']
    PERDA_TEMPO = VIGA['PERDA_TEMPO']
    U = VIGA['U']
    F_PK = VIGA['F_PK']
    F_YK = VIGA['F_YK']
    LAMBA_SIG = VIGA['LAMBA_SIG']
    TIPO_FIO_CORD_BAR = VIGA['TIPO_FIO_CORD_BAR']
    TIPO_PROT = VIGA['TIPO_PROT']
    TIPO_ACO = VIGA['TIPO_ACO']
    PHO_C = VIGA['PHO_C']
    F_CK = VIGA['F_CK']
    CIMENTO = VIGA['CIMENTO']
    AGREGADO = VIGA['AGREGADO']
    ABAT = VIGA['ABAT']
    G_2K = VIGA['G_2K']
    Q_1K = VIGA['Q_1K']
    PSI_1 = VIGA['PSI_1']
    PSI_2 = VIGA['PSI_2'] 
    GAMMA_F1 = VIGA['GAMMA_F1']
    GAMMA_F2 = VIGA['GAMMA_F2']
    GAMMA_S = VIGA['GAMMA_S']
    E_SCP = VIGA['E_SCP']
    ETA_1 = VIGA['ETA_1']
    ETA_2 = VIGA['ETA_2'] 
    E_PPROPORCAO = VIGA['E_PPROPORCAO'] 
    A_BAR = VIGA['A_BAR']
    # Pré-processamento de alguns dados
    H = H_W + H_SI + H_II + H_FS + H_FI
    MU_AR = H * 2 + B_W * 2 + (B_FS - B_W) * 2 +  (B_FI - B_W) * 2
    PERDA_TOTAL = PERDA_INICIAL + PERDA_TEMPO
    E_P = E_PPROPORCAO * H
    G = []
    # Determinação das propriedades da seção tipo I
    [A_C, I_C, Y_SUP, Y_INF, W_SUP, W_INF] = PROP_GEOMETRICA_VIGA_I(H, B_FS, B_FI, B_W, H_FS, H_FI, H_SI, H_II)
    if IMPRESSAO:
        print('Propriedades da seção transversal tipo I com abas inclinadas')
        print('A_C:   ', '%+10.5e' % A_C, 'm²')
        print('I_C:   ', '%+10.5e' % I_C, 'm^4')
        print('Y_SUP: ', '%+10.5e' % Y_SUP, 'm')
        print('Y_INF: ', '%+10.5e' % Y_INF, 'm')
        print('W_SUP: ', '%+10.5e' % W_SUP, 'm³')
        print('W_INF: ', '%+10.5e' % W_INF, 'm³')
        print('E_P:   ', '%+10.5e' % E_P, 'm')
        print('\n') 
        print('Propriedades do material')  
    # Propriedades do material em todos as etapas construtivas
    F_CKJ = []; F_CTMJ = []; F_CTKINFJ = []; F_CTKSUPJ = []; E_CIJ = []; E_CSJ = [];
    for I_COUNT in range(len(TEMPO_CONC)):
        TEMPO = TEMPO_CONC[I_COUNT]
        [F_CKJJ, F_CTM, F_CTKINF, F_CTKSUP, E_CI, E_CS] = PROP_MATERIAL(F_CK, TEMPO, CIMENTO, AGREGADO)
        F_CKJ.append(F_CKJJ); F_CTMJ.append(F_CTM); F_CTKINFJ.append(F_CTKINF)
        F_CTKSUPJ.append(F_CTKSUP); E_CIJ.append(E_CI), E_CSJ.append(E_CS) 
        if IMPRESSAO:
            print('Idade:    ', TEMPO, 'dias')
            print('F_CKJ:    ', '%+10.5e' % F_CKJJ, 'kN/m²')
            print('F_CTM:    ', '%+10.5e' % F_CTM, 'kN/m²')
            print('F_CTKINF: ', '%+10.5e' % F_CTKINF, 'kN/m²')
            print('F_CTKSUP: ', '%+10.5e' % F_CTKSUP, 'kN/m²')
            print('E_CI:     ', '%+10.5e' % E_CI, 'kN/m²')
            print('E_CS:     ', '%+10.5e' % E_CS, 'kN/m²')
            print('\n')
    # Tensão inicial
    SIGMA_PI0 = TENSAO_INICIAL(TIPO_PROT, TIPO_ACO, F_PK, F_YK)
    SIGMA_PI0 *= LAMBA_SIG
    if IMPRESSAO:
        print('Tensão inicial na armadura')
        print('SIGMA_PI0: ', '%+10.5e' % SIGMA_PI0, 'kN/m²')
        print('\n')
        print('Determinação da perda de protensão até sua estabilização')
    ERRO = 1000
    CONT = 0
    while ERRO > (1 / 1E4):
            if IMPRESSAO:
                print('Tentativa de definição da perda :', CONT, ' Perda Total :', '%6.3f' % PERDA_TOTAL, ' Erro: ', '%-6.3e' % ERRO)
            # Cálculo da tensão de protensão para perdas inicial e final
            SIGMA_PIINI = SIGMA_PI0 - SIGMA_PI0 * (PERDA_INICIAL / 100)
            PERDA_TOTAL = PERDA_INICIAL + PERDA_TEMPO
            SIGMA_PIINF = SIGMA_PI0 - SIGMA_PI0 * (PERDA_TOTAL / 100)
            # Determinação do comprimento de transferência médio
            L_PINI = COMPRIMENTO_TRANSFERENCIA(PHI_L, F_YK, F_CTKINFJ[0], ETA_1, ETA_2, SIGMA_PIINI, H)
            L_PINF = COMPRIMENTO_TRANSFERENCIA(PHI_L, F_YK, F_CTKINFJ[-1], ETA_1, ETA_2, SIGMA_PIINF, H)
            L_P = np.mean([L_PINI, L_PINF])
            # Carregamento devido ao peso próprio
            G_1K = A_C * PHO_C
            # Momento máximo (meio do vão), momento apoios (no comp. de transferência) e cortante máximo (apoios)
            M_MVG1K, M_APG1K, V_APG1K = ESFORCOS(G_1K, L, L_P)
            M_MVG2K, M_APG2K, V_APG2K = ESFORCOS(G_2K, L, L_P)
            M_MVQ1K, M_APQ1K, V_APQ1K = ESFORCOS(Q_1K, L, L_P)
            # Avaliação das tensões no ELS: seções do apoio
            if (E_P + 50 / 1E3) > Y_INF:
                Y_I = Y_INF
            else:
                Y_I = (E_P + 50 / 1E3)
            # Avaliação das tensões no ELS: seções do apoio e meio do vão
            A_SCPINICIAL0 = ARMADURA_ASCP_ELS(A_C, I_C, Y_I, E_P, PSI_1, PSI_2, M_APG1K, M_APG2K, 0, M_APQ1K, SIGMA_PIINF, F_CTKINFJ[-1], FATOR_SEC)
            A_SCPINICIAL1 = ARMADURA_ASCP_ELS(A_C, I_C, Y_I, E_P, PSI_1, PSI_2, M_MVG1K, M_MVG2K, 0, M_MVQ1K, SIGMA_PIINF, F_CTKINFJ[-1], FATOR_SEC)
            # Seleção do maior valor de área de aço (pior situação)
            A_SCP = max(A_SCPINICIAL0, A_SCPINICIAL1)
            N_BAR = math.ceil(A_SCP / A_BAR)
            A_SCP = N_BAR * A_BAR
            # Reavaliando as perdas de protensão
            # Perdas iniciais
            P_I0 = SIGMA_PI0 * A_SCP
            DELTA1, P_I1, SIGMA_PI1 = PERDA_DESLIZAMENTO_ANCORAGEM(P_I0, SIGMA_PI0, A_SCP, L_PISTA, DELTA_ANC, E_SCP)
            DELTA2, P_I2, SIGMA_PI2, PSI2 = PERDA_RELAXACAO_ARMADURA(P_I1, SIGMA_PI1, 0, TEMPO_ACO[0], TEMP, F_PK, A_SCP, TIPO_FIO_CORD_BAR, TIPO_ACO)
            DELTA3, P_I3, SIGMA_PI3 = PERDA_DEFORMACAO_CONCRETO(E_SCP, E_CIJ[0], P_I2, SIGMA_PI2, A_SCP, A_C, I_C, E_P, M_MVG1K)
            DELTA_INI = SIGMA_PI0 - SIGMA_PI3 
            PERDA_INICIALAUX = (DELTA_INI / SIGMA_PI0) * 100
            SIGMA_PIINI = SIGMA_PI0 - SIGMA_PI0 * (PERDA_INICIALAUX / 100)
            P_IINI = P_I0 - P_I0 * (PERDA_INICIALAUX / 100)
            # Perdas progressivas
            DELTA4, P_I4, SIGMA_PI4 = PERDA_RETRACAO_CONCRETO(P_IINI, SIGMA_PIINI, A_SCP, U, ABAT, A_C, MU_AR, TEMPO_CONC[0], TEMPO_CONC[-1], TEMP, 'RETRACAO', 'RAPIDO', E_SCP)
            DELTA5, P_I5, SIGMA_PI5, PSI5 = PERDA_RELAXACAO_ARMADURA(P_IINI, SIGMA_PIINI, TEMPO_ACO[0], TEMPO_ACO[-1], TEMP, F_PK, A_SCP, TIPO_FIO_CORD_BAR, TIPO_ACO)
            PHI_GP = PHI_FLUENCIA(F_CKJ[0], F_CK, U, A_C, MU_AR, ABAT, TEMPO_CONC[0], TEMPO_CONC[-1], TEMP, 'FLUENCIA', 'RAPIDO')
            PHI_G2 = PHI_FLUENCIA(F_CKJ[-3], F_CK, U, A_C, MU_AR, ABAT, TEMPO_CONC[-3], TEMPO_CONC[-1], TEMP, 'FLUENCIA', 'RAPIDO')
            PHI_Q1 = PHI_FLUENCIA(F_CKJ[-2], F_CK, U, A_C, MU_AR, ABAT, TEMPO_CONC[-2], TEMPO_CONC[-1], TEMP, 'FLUENCIA', 'RAPIDO')
            PHI = [PHI_GP, PHI_G2, PHI_Q1]
            M_PFLU = P_IINI * E_P
            SIGMA_GP = P_IINI / A_C + ((M_PFLU - M_MVG1K) / I_C) * E_P
            SIGMA_G2 = - (M_MVG2K * E_P) / I_C
            SIGMA_Q1 = - (PSI_2 * M_MVQ1K * E_P) / I_C
            SIGMA_CABO = [SIGMA_GP, SIGMA_G2, SIGMA_Q1]
            DELTA6, P_I6, SIGMA_PI6 = PERDA_POR_FLUENCIA_NO_CONCRETO(P_IINI, SIGMA_PIINI, A_SCP, PHI, E_SCP, E_CIJ[-1], SIGMA_CABO)
            DELTA_RET =  SIGMA_PIINI - SIGMA_PI4; DELTA_FLU = SIGMA_PIINI - SIGMA_PI6
            DELTA7, P_I7, SIGMA_PI7 = INTERACAO_ENTRE_PERDAS_PROGRESSIVAS(E_P, A_C, I_C, E_SCP, E_CIJ[-1], DELTA_RET, DELTA_FLU, P_IINI, SIGMA_PIINI, PSI5, A_SCP, PHI_GP)
            PERDA_TOTALAUX = ((SIGMA_PI0 - SIGMA_PI7) / SIGMA_PI0) * 100 
            ERRO = np.abs(PERDA_TOTALAUX - PERDA_TOTAL) / PERDA_TOTAL
            CONT += 1
            PERDA_INICIAL = PERDA_INICIALAUX
            PERDA_TOTAL = PERDA_TOTALAUX
            PERDA_TEMPO = PERDA_TOTAL - PERDA_INICIAL
    if IMPRESSAO:
        print('\n')
        print('Comprimento de transferência considerando as perdas iniciais')
        print('L_P: ', '%+10.5e' % L_P, 'm')
        print('\n')
        print('Esforços devido aos carregametos G e Q (M = kN.m, V = kN)')
        print('MV = Meio do vão, AP = Apoio')
        print('M_G1MV:  ', '%+10.5e' % M_MVG1K, '   M_G1AP:  ', '%+10.5e' % M_APG1K,  '   V_G1AP:  ', '%+10.5e' % V_APG1K)
        print('M_G2MV:  ', '%+10.5e' % M_MVG2K, '   M_G2AP:  ', '%+10.5e' % M_APG2K,  '   V_G2AP:  ', '%+10.5e' % V_APG2K)
        print('M_Q1MV:  ', '%+10.5e' % M_MVQ1K, '   M_Q1AP:  ', '%+10.5e' % M_APQ1K,  '   V_Q1AP:  ', '%+10.5e' % V_APQ1K)
        print('\n')
        print('Sugestão de armadura ')
        print('A_SCPINICIAL >=  ', '%+10.5e' % A_SCPINICIAL0, 'm²', '  Sugestão de armadura respeitando as condições nos apoios')
        print('A_SCPINICIAL >=  ', '%+10.5e' % A_SCPINICIAL1, 'm²', '  Sugestão de armadura respeitando as condições no meio do vão')
        print('A_SCP        ==  ', '%+10.5e' % A_SCP, 'm²', '  Armadura inicial escolhida')
        print('\n')
        print('Perdas de protensão após definição')
        print('Perda inicial     =  ', '%+10.5e' % PERDA_INICIAL, '%')
        print('Perda progressiva =  ', '%+10.5e' % PERDA_TEMPO, '%') 
        print('Perda total       =  ', '%+10.5e' % PERDA_TOTAL, '%')
        print('\n') 
    # Carga inicial de protensão após definição da armadura
    SIGMA_PI1 = SIGMA_PI0 - SIGMA_PI0 * (PERDA_INICIAL / 100)
    P_I1 = SIGMA_PI1 * A_SCP
    if IMPRESSAO:
        print('Verificações ELU no ato da protensão')
        print('Parâmetros da protensão')
        print('SIGMA_PI1:     ' , '%+10.5e' % SIGMA_PI1, 'kN/m²')
        print('P_I1:          ' , '%+10.5e' % P_I1, 'kN')
        print('\n')
    # Valores máximos das tensões (tração e compressão) para verificação das tensões nas faces
    SIGMA_TRACMAX = 1.20 * F_CTMJ[0]
    SIGMA_COMPMAX = 0.70 * F_CKJ[0]
    if IMPRESSAO:
        print('Valores máximos das tensões ')
        print('SIGMA_TRACMAX: ' , '%+10.5e' % SIGMA_TRACMAX, 'kN/m²')
        print('SIGMA_COMPMAX: ' , '%+10.5e' % SIGMA_COMPMAX, 'kN/m²')
        print('\n')
    # Verificação tensões normais no ato da protensão no apoio
    [SIGMA_INF, SIGMA_SUP] = TENSOES_NORMAIS(P_I1, A_C, E_P, W_INF, W_SUP, 1, 1, 0, 0, 0, 0, 0, M_APG1K, 0, 0, 0, 0)
    [G_0, G_1] = VERIFICA_TENSAO_NORMAL_ATO_PROTENSÃO(SIGMA_INF, SIGMA_SUP, SIGMA_TRACMAX, SIGMA_COMPMAX)
    G.append(G_0); G.append(G_1)
    if IMPRESSAO:
        print('Verificação das tensões normais nos bordos no apoio ')
        print('Bordo inferior:', '%+10.5e' % SIGMA_INF, 'kN/m²', '// Eq. Estado Limite:', '%+10.5e' % G_0)
        print('Bordo superior:', '%+10.5e' % SIGMA_SUP, 'kN/m²', '// Eq. Estado Limite:', '%+10.5e' % G_1)
        print('\n')
    # Verificação tensões normais no ato da protensão no meio do vão
    [SIGMA_INF, SIGMA_SUP] = TENSOES_NORMAIS(P_I1, A_C, E_P, W_INF, W_SUP, 1, 1, 0, 0, 0, 0, 0, M_MVG1K, 0, 0, 0, 0)
    [G_2, G_3] = VERIFICA_TENSAO_NORMAL_ATO_PROTENSÃO(SIGMA_INF, SIGMA_SUP, SIGMA_TRACMAX, SIGMA_COMPMAX)
    G.append(G_2); G.append(G_3)
    if IMPRESSAO:
        print('Verificação das tensões normais nos bordos no meio do vão ')
        print('Bordo inferior:', '%+10.5e' % SIGMA_INF, 'kN/m²', '// Eq. Estado Limite:', '%+10.5e' % G_2)
        print('Bordo superior:', '%+10.5e' % SIGMA_SUP, 'kN/m²', '// Eq. Estado Limite:', '%+10.5e' % G_3)
        print('\n')
    SIGMA_PIINF = SIGMA_PI0 - SIGMA_PI0 * (PERDA_TOTAL / 100)
    # Momento de cálculo
    M_SD = (M_MVG1K) * GAMMA_F1 + (M_MVG2K + M_MVQ1K) * GAMMA_F2
    # Altura útil
    D = Y_SUP + E_P
    # Propriedades do aço
    F_PD = F_PK / GAMMA_S
    F_YD = F_YK / GAMMA_S
    EPSILON_Y = F_YD / E_SCP
    # Armadura necessária
    X, EPSILON_S, EPSILON_C, Z, A_SCPNEC = AREA_ACO_LONGITUDINAL_CP_T(M_SD, F_CKJ[-1], B_W, B_FS, H_FS, D, E_SCP, SIGMA_PIINF, 35/1000, EPSILON_Y, F_PD, F_YD)
    G_4 = VERIFICA_ARMADURA_FLEXAO(A_SCP, A_SCPNEC)
    G.append(G_4)
    if IMPRESSAO:
        print('Verificação da armadura necessária')
        print('M_SD:      ', '%+10.5e' % M_SD, 'kN.m')
        print('D:         ', '%+10.5e' % D, 'm')
        print('F_PD:      ', '%+10.5e' % F_PD, 'kN/m²')
        print('F_YD:      ', '%+10.5e' % F_YD, 'kN/m²')
        print('EPSILON_Y: ', '%+10.5e' % EPSILON_Y)
        print('X:         ',  '%+10.5e' % X, 'm')
        print('EPSILON_S: ',  '%+10.5e' % EPSILON_S)
        print('EPSILON_C: ',  '%+10.5e' % EPSILON_C)
        print('Z:         ',  '%+10.5e' % Z, 'm')
        print('A_SCPNEC:  ',  '%+10.5e' % A_SCPNEC, 'm²', '// Eq. Estado Limite:', '%+10.5e' % G_4)
        print('\n')
    # Verificação da biela comprimida
    V_SD = (V_APG1K) * GAMMA_F1 + (V_APG2K + V_APQ1K) * GAMMA_F2
    V_RD2 = RESISTENCIA_BIELA_COMPRIMIDA(F_CKJ[-1], B_W, D)
    G_5 = VERIFICA_BIELA_COMPRIMIDA(V_SD, V_RD2)
    G.append(G_5)
    # Cálculo da armadura necessária
    P_IINF = SIGMA_PIINF * A_SCP
    V_C, V_SW, A_SW = AREA_ACO_TRANSVERSAL_MODELO_I(0, P_IINF, V_SD, F_CTKINFJ[-1], B_W, D, 'CP', W_INF, A_C, E_P, M_SD, F_CTMJ[-1], 500E3)
    if IMPRESSAO:
        print('Verificação da biela de compressão')
        print('V_SD:  ',   '%+10.5e' % V_SD, 'kN')
        print('V_RD2: ',   '%+10.5e' % V_RD2, 'kN', '// Eq. Estado Limite:', '%+10.5e' % G_5)
        print('V_C:   ',   '%+10.5e' % V_C, 'kN')
        print('V_SW:  ',   '%+10.5e' % V_SW, 'kN')
        print('A_SW:  ',   '%+10.5e' % A_SW, 'm²/m')
        print('\n')
    # Determinação das propriedades no Estádio I
    ALPHA_MOD = E_SCP / E_CSJ[-1]
    #A_CI, X_I, I_I = PROP_GEOMETRICA_ESTADIO_I(H, B_FS, B_W, H_FS, A_SCP, ALPHA_MOD, D)
    M_R = MOMENTO_RESISTENTE('I', F_CTMJ[-1], H, Y_SUP, I_C, P_IINF, A_C, W_INF, E_P)
    # Determinação das propriedades no Estádio II
    X_II, I_II = PROP_GEOMETRICA_ESTADIO_II(H, B_FS, B_W, H_FS, A_SCP, 0, ALPHA_MOD, D, 0)
    # Momento atuante
    M_SER = (M_MVG1K + M_MVG2K) + (M_MVQ1K) * PSI_2
    # Inércia e rigidez equivalente da peça
    if M_SER > M_R:
        I_M = INERCIA_BRANSON(M_R, M_SER, I_C, I_II)
        I_CNOVO = I_M
        ESTADIO_CALC = 'Peça deverá ser verificada no Estádio II'
    else:
        I_CNOVO  = I_C
        ESTADIO_CALC = 'Peça deverá ser verificada no Estádio I'
    # Rigidez em cada uma das etapas analisadas
    EI0 = E_CSJ[0] * I_CNOVO
    EI1 = E_CSJ[-1] * I_CNOVO
    # Cálculo das flechas totais
    # Flecha na fabircação
    M_PI = P_I1 * E_P
    A_PI = (- 2 * M_PI * (L ** 2)) / (9 * np.sqrt(3) * EI0)
    A_G1 = (5 * G_1K * (L ** 4) / (384 * EI0))
    A_FABRICA = np.abs(A_G1 + A_PI)
    # Flecha no serviço
    A_G2 = (5 * G_2K * (L ** 4) / (384 * EI1))
    A_Q1 = (5 * (PSI_1 * Q_1K) * (L ** 4) / (384 * EI1))
    A_SERVICO = ((PHI_GP + 1) * (A_G1 + A_PI)) + A_G2 * (PHI_G2 + 1) + A_Q1
    # Verificação flecha na fabricação
    G_6 = VERIFICA_FLECHA(A_FABRICA, L / 1000)
    G_7 = VERIFICA_FLECHA(A_SERVICO, L / 250)
    if IMPRESSAO:
        print("Verificação da flecha")
        print('M_R:'     , '%+10.5e' % M_R, 'kN.m')      
        print('X_II:'    , '%+10.5e' % X_II, 'm')
        print('I_II:'    , '%+10.5e' % I_II, 'm^4')
        print('M_SER:'   , '%+10.5e' % M_SER, 'kN.m')
        print(ESTADIO_CALC)
        print('I_EQ:'    , '%+10.5e' % I_CNOVO, 'm^4')
        print('E.I[INI]:', '%+10.5e' % EI0, 'kN.m²')
        print('E.I[FIM]:', '%+10.5e' % EI1, 'kN.m²')
        print('A_P:'     , '%+10.5e' % A_PI, 'm')
        print('A_G1:'    , '%+10.5e' % A_G1, 'm')
        print('A_G2:'    , '%+10.5e' % A_G2, 'm')
        print('A_Q1:'    , '%+10.5e' % A_Q1, 'm')
        print("Flecha total na fabricação      = ", A_FABRICA, 'm', '// Eq. Estado Limite:', G_6)
        print("Flecha total no serviço         = ", A_SERVICO, 'm', '// Eq. Estado Limite:', G_7)
    return G, A_C, A_SCP

def EPSILON_COEFICIENTE(T):
    """
    Esta função calcula o fator da flecha.

    Entrada:
    T         | Tempo                                              | mês  | float
    
    Saída:
    EPSILON   | Fator da flecha                                    |      | float
    """
    if T < 70:
        EPSILON = 0.68 * (0.996 ** T) * (T ** 0.32)
    else:
        EPSILON = 2
    return EPSILON

def TOTAL_DISPLACEMENT(FLECHA_INICIAL, A_ST, B_W, D, T_0, T_1):
    """
    Esta função calcula o valor da flecha.

    Entrada:
    FLECHA_INICIAL | Valor da flecha                                    | m    | float
    A_ST           | Area de aço na seção comprimida                    | m²   | float
    B_W            | Base de alma da viga                               | m    | float
    D              | Altura útil da seção                               | m    | float
    T_0            | Tempo inicial                                      | mês  | float
    T_1            | Tempo final                                        | mês  | float
    
    Saída:
    PHO_L          | $$                                                 |      | float
    ALPHA_F        | ###############                                    |      | float
    FLECHA_TOTAL   | Flecha da viga                                     | m    | float
    """
    EPSILON_INITITAL = EPSILON_COEFFICIENT(T_INITIAL)
    EPSILON_END = EPSILON_COEFFICIENT(T_END)
    DELTA_EPSILON = EPSILON_END - EPSILON_INITITAL
    PHO_L = A_ST / (B_W * D)
    ALPHA_F = DELTA_EPSILON / (1 + PHO_L)
    FLECHA_TOTAL = DELTA_INITIAL * (1 + ALPHA_F)
    return PHO_L, ALPHA_F, FLECHA_TOTAL  