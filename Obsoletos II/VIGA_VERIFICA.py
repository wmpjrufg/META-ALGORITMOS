################################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR,                  ENG. CIVIL / PROF (UFCAT)
# MATHEUS HENRIQUE MORATO DE MORAES                    ENG. CIVIL / PROF (UFCAT)
################################################################################

################################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIOTECA DE CÁLCULO DE VIGAS PROTENDIDAS DESENVOLVIDA PELO GRUPO DE PESQUISA 
# E ESTUDOS EM ENGENHARIA (GPEE)
################################################################################


################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import numpy as np

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE
from VIGA_PREPRO import *
from PERDAS import *

def VERIFICACAO_VIGA(VIGA):
    # Atribuição de dados
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
    # Pré-processamento de alguns dados
    H = H_W + H_SI + H_II + H_FS + H_FI
    MU_AR = H * 2 + B_W * 2 + (B_FS - B_W) * 2 +  (B_FI - B_W) * 2
    PERDA_TOTAL = PERDA_INICIAL + PERDA_TEMPO
    E_P = E_PPROPORCAO * H
    G = []
    # Determinação das propriedades da seção tipo I
    [A_C, I_C, Y_SUP, Y_INF, W_SUP, W_INF] = PROP_GEOMETRICA_I(H, B_FS, B_FI, B_W, H_FS, H_FI, H_SI, H_II)
    #"""
    print('Propriedades da seção transversal tipo I com abas inclinadas')
    print('A_C:   ', '%+10.5e' % A_C, 'm²')
    print('I_C:   ', '%+10.5e' % I_C, 'm^4')
    print('Y_SUP: ', '%+10.5e' % Y_SUP, 'm')
    print('Y_INF: ', '%+10.5e' % Y_INF, 'm')
    print('W_SUP: ', '%+10.5e' % W_SUP, 'm³')
    print('W_INF: ', '%+10.5e' % W_INF, 'm³')
    print('E_P:   ', '%+10.5e' % E_P, 'm')
    print('\n')
    #"""
    # Propriedades do material em todos as etapas construtivas
    F_CKJ = []; F_CTMJ = []; F_CTKINFJ = []; F_CTKSUPJ = []; E_CIJ = []; E_CSJ = [];
    #"""
    print('Propriedades do material')
    #"""
    for I_COUNT in range(len(TEMPO_CONC)):
        TEMPO = TEMPO_CONC[I_COUNT]
        [F_CKJJ, F_CTM, F_CTKINF, F_CTKSUP, E_CI, E_CS] = PROP_MATERIAL(F_CK, TEMPO, CIMENTO, AGREGADO)
        F_CKJ.append(F_CKJJ); F_CTMJ.append(F_CTM); F_CTKINFJ.append(F_CTKINF)
        F_CTKSUPJ.append(F_CTKSUP); E_CIJ.append(E_CI), E_CSJ.append(E_CS) 
        """
        print('Idade:    ', TEMPO, 'dias')
        print('F_CKJ:    ', '%+10.5e' % F_CKJJ, 'kN/m²')
        print('F_CTM:    ', '%+10.5e' % F_CTM, 'kN/m²')
        print('F_CTKINF: ', '%+10.5e' % F_CTKINF, 'kN/m²')
        print('F_CTKSUP: ', '%+10.5e' % F_CTKSUP, 'kN/m²')
        print('E_CI:     ', '%+10.5e' % E_CI, 'kN/m²')
        print('E_CS:     ', '%+10.5e' % E_CS, 'kN/m²')
        print('\n')
        """
    # Tensão inicial
    SIGMA_PI0 = TENSAO_INICIAL(TIPO_PROT, TIPO_ACO, F_PK, F_YK)
    SIGMA_PI0 *= LAMBA_SIG
    """
    print('Tensão inicial na armadura')
    print('SIGMA_PI0: ', '%+10.5e' % SIGMA_PI0, 'kN/m²')
    """
    ERRO = 1000
    CONT = 0
    while ERRO > (1 / 1E3):
            """
            print('Tentativa de definição da perda :', CONT, ' Perda Total :', PERDA_TOTAL, ' Erro: ', ERRO)
            """
            # Cálculo da tensão de protensão para perdas inicial e final
            SIGMA_PIINI = SIGMA_PI0 - SIGMA_PI0 * (PERDA_INICIAL / 100)
            PERDA_TOTAL = PERDA_INICIAL + PERDA_TEMPO
            SIGMA_PIINF = SIGMA_PI0 - SIGMA_PI0 * (PERDA_TOTAL / 100)
            # Determinação do comprimento de transferência médio
            L_PINI = COMPRIMENTO_TRANSFERENCIA(PHI_L, F_YK, F_CTKINFJ[0], ETA_1, ETA_2, SIGMA_PIINI, H)
            L_PINF = COMPRIMENTO_TRANSFERENCIA(PHI_L, F_YK, F_CTKINFJ[4], ETA_1, ETA_2, SIGMA_PIINF, H)
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
            A_SCPINICIAL0 = ARMADURA_ASCP_ELS(A_C, I_C, Y_I, E_P, PSI_1, PSI_2, M_APG1K, M_APG2K, 0, M_APQ1K, SIGMA_PIINF, F_CTKINFJ[5], FATOR_SEC)
            # Avaliação das tensões no ELS: seções do meio do vão
            A_SCPINICIAL1 = ARMADURA_ASCP_ELS(A_C, I_C, Y_I, E_P, PSI_1, PSI_2, M_MVG1K, M_MVG2K, 0, M_MVQ1K, SIGMA_PIINF, F_CTKINFJ[5], FATOR_SEC)
            # Seleção do maior valor de área de aço (pior situação)
            A_SCP = max(A_SCPINICIAL0, A_SCPINICIAL1)
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
            DELTA4, P_I4, SIGMA_PI4 = PERDA_RETRACAO_CONCRETO(P_IINI, SIGMA_PIINI, A_SCP, U, ABAT, A_C, MU_AR, TEMPO_CONC[0], TEMPO_CONC[5], TEMP, 'RETRACAO', 'RAPIDO', E_SCP)
            DELTA5, P_I5, SIGMA_PI5, PSI5 = PERDA_RELAXACAO_ARMADURA(P_IINI, SIGMA_PIINI, TEMPO_ACO[0], TEMPO_ACO[5], TEMP, F_PK, A_SCP, TIPO_FIO_CORD_BAR, TIPO_ACO)
            PHI_GP = PHI_FLUENCIA(F_CKJ[0], F_CK, U, A_C, MU_AR, ABAT, TEMPO_CONC[0], TEMPO_CONC[5], TEMP, 'FLUENCIA', 'RAPIDO')
            PHI_G2 = PHI_FLUENCIA(F_CKJ[3], F_CK, U, A_C, MU_AR, ABAT, TEMPO_CONC[3], TEMPO_CONC[5], TEMP, 'FLUENCIA', 'RAPIDO')
            PHI_Q1 = PHI_FLUENCIA(F_CKJ[4], F_CK, U, A_C, MU_AR, ABAT, TEMPO_CONC[4], TEMPO_CONC[5], TEMP, 'FLUENCIA', 'RAPIDO')
            PHI = [PHI_GP, PHI_G2, PHI_Q1]
            SIGMA_GP = P_IINI / A_C + (((P_IINI * E_P) - M_MVG1K) / I_C) * E_P
            SIGMA_G2 = - (M_MVG2K * E_P) / I_C
            SIGMA_Q1 = - (PSI_2 * M_MVQ1K * E_P) / I_C
            SIGMA_CABO = [SIGMA_GP, SIGMA_G2, SIGMA_Q1]
            DELTA6, P_I6, SIGMA_PI6 = PERDA_POR_FLUENCIA_NO_CONCRETO(P_IINI, SIGMA_PIINI, A_SCP, PHI, E_SCP, E_CIJ[5], SIGMA_CABO)
            DELTA_RET =  SIGMA_PIINI - SIGMA_PI4; DELTA_FLU = SIGMA_PIINI - SIGMA_PI6
            DELTA7, P_I7, SIGMA_PI7 = INTERACAO_ENTRE_PERDAS_PROGRESSIVAS(E_P, A_C, I_C, E_SCP, E_CIJ[5], DELTA_RET, DELTA_FLU, P_IINI, SIGMA_PIINI, PSI5, A_SCP, PHI_GP)
            DELTA_INF = SIGMA_PIINI - SIGMA_PI7
            PERDA_TEMPOAUX = (DELTA_INF / SIGMA_PI0) * 100
            PERDA_TOTALAUX = PERDA_INICIALAUX + PERDA_TEMPOAUX
            ERRO = np.abs(PERDA_TOTALAUX - PERDA_TOTAL) / PERDA_TOTAL
            CONT += 1
            PERDA_INICIAL = PERDA_INICIALAUX
            PERDA_TEMPO = PERDA_TEMPOAUX
            PERDA_TOTAL = PERDA_TOTALAUX
    """
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
    print('Perda inicial')
    print('% =  ', '%+10.5e' % PERDA_INICIAL)
    print('Perda progressiva')
    print('% =  ', '%+10.5e' % PERDA_TEMPO) 
    print('Perda total')
    print('% =  ', '%+10.5e' % PERDA_TOTAL) 
    """
    # Carga inicial de protensão após definição da armadura
    SIGMA_PI1 = SIGMA_PI0 - SIGMA_PI0 * (PERDA_INICIAL / 100)
    P_I1 = SIGMA_PI1 * A_SCP
    """
    print('Verificações ELU no ato da protensão')
    print('Parâmetros da protensão')
    print('SIGMA_PI1:     ' , '%+10.5e' % SIGMA_PI1, 'kN/m²')
    print('P_I1:          ' , '%+10.5e' % P_I1, 'kN')
    print('\n')
    """
    # Valores máximos das tensões (tração e compressão)
    SIGMA_TRACMAX = 1.20 * F_CTMJ[0]
    SIGMA_COMPMAX = 0.70 * F_CKJ[0]
    """
    print('Valores máximos das tensões ')
    print('SIGMA_TRACMAX: ' , '%+10.5e' % SIGMA_TRACMAX, 'kN/m²')
    print('SIGMA_COMPMAX: ' , '%+10.5e' % SIGMA_COMPMAX, 'kN/m²')
    print('\n')
    """
    # Verificação tensões normais no ato da protensão no apoio
    [SIGMA_INF, SIGMA_SUP] = TENSOES_NORMAIS(P_I1, A_C, E_P, W_INF, W_SUP, 1, 1, 0, 0, 0, 0, 0, M_APG1K, 0, 0, 0, 0)
    [G_0, G_1] = VERIFICA_TENSAO_NORMAL_ATO_PROTENSÃO(SIGMA_INF, SIGMA_SUP, SIGMA_TRACMAX, SIGMA_COMPMAX)
    G.append(G_0); G.append(G_1)
    """
    print('Verificação das tensões normais nos bordos no apoio ')
    print('Bordo inferior:', '%+10.5e' % SIGMA_INF, 'kN/m²', '// Eq. Estado Limite:', '%+10.5e' % G_0)
    print('Bordo superior:', '%+10.5e' % SIGMA_SUP, 'kN/m²', '// Eq. Estado Limite:', '%+10.5e' % G_1)
    print('\n')
    """
    # Verificação tensões normais no ato da protensão no meio do vão
    [SIGMA_INF, SIGMA_SUP] = TENSOES_NORMAIS(P_I1, A_C, E_P, W_INF, W_SUP, 1, 1, 0, 0, 0, 0, 0, M_MVG1K, 0, 0, 0, 0)
    [G_2, G_3] = VERIFICA_TENSAO_NORMAL_ATO_PROTENSÃO(SIGMA_INF, SIGMA_SUP, SIGMA_TRACMAX, SIGMA_COMPMAX)
    G.append(G_2); G.append(G_3)
    """
    print('Verificação das tensões normais nos bordos no meio do vão ')
    print('Bordo inferior:', '%+10.5e' % SIGMA_INF, 'kN/m²', '// Eq. Estado Limite:', '%+10.5e' % G_2)
    print('Bordo superior:', '%+10.5e' % SIGMA_SUP, 'kN/m²', '// Eq. Estado Limite:', '%+10.5e' % G_3)
    print('\n')
    """
    SIGMA_PIINF = SIGMA_PI0 - SIGMA_PI0 * (PERDA_TOTAL / 100)
    # Momento de cálculo
    M_SD = (M_MVG1K) * GAMMA_F1 + (M_MVG2K + M_MVQ1K) * GAMMA_F2
    """
    print('Verificação da armadura necessária')
    print('Momento de cálculo')
    print('M_SD: ', '%+10.5e' % M_SD, 'kN.m')
    print('\n')
    """
    # Altura útil
    D = Y_SUP + E_P
    """
    print('Altura útil')
    print('D: ',   '%+10.5e' % D, 'm')
    print('\n')
    """
    # Propriedades do aço
    F_PD = F_PK / GAMMA_S
    F_YD = F_YK / GAMMA_S
    EPSILON_Y = F_YD / E_SCP
    """
    print('Propriedades do aço')
    print('F_PD:      ',   '%+10.5e' % F_PD, 'kN/m²')
    print('F_YD:      ',   '%+10.5e' % F_YD, 'kN/m²')
    print('EPSILON_Y: ',   '%+10.5e' % EPSILON_Y)
    print('\n')
    """
    # Armadura necessária
    X, EPSILON_S, EPSILON_C, Z, A_SCPNEC = AREA_ACO_LONGITUDINAL_CP_T(M_SD, F_CKJ[5], B_W, B_FS, H_FS, D, E_SCP, SIGMA_PIINF, 35/1000, EPSILON_Y, F_PD, F_YD)
    G_4 = VERIFICA_ARMADURA_FLEXAO(A_SCP, A_SCPNEC)
    G.append(G_4)
    """print('Armadura necessária')
    print('X:         ',   '%+10.5e' % X, 'm')
    print('EPSILON_S: ',   '%+10.5e' % EPSILON_S)
    print('EPSILON_C: ',   '%+10.5e' % EPSILON_C)
    print('Z:         ',   '%+10.5e' % Z, 'm')
    print('A_SCPNEC:  ',   '%+10.5e' % A_SCPNEC, 'm²', '// Eq. Estado Limite:', '%+10.5e' % G_4)
    print('\n')
    """
    # Verificação da biela comprimida
    V_SD = (V_APG1K) * GAMMA_F1 + (V_APG2K + V_APQ1K) * GAMMA_F2
    V_RD2 = RESISTENCIA_BIELA_COMPRIMIDA(F_CKJ[5], B_W, D)
    G_5 = VERIFICA_BIELA_COMPRIMIDA(V_SD, V_RD2)
    G.append(G_5)
    """
    print('Verificação da biela de compressão')
    print('Esforço cortante solicitante')
    print('V_SD:         ',   '%+10.5e' % V_SD, 'kN')
    print('\n')
    print('Resistência da biela de compressão')
    print('V_RD2:        ',   '%+10.5e' % V_RD2, 'kN', '// Eq. Estado Limite:', '%+10.5e' % G_5)
    print('\n')
    """
    # Cálculo da armadura necessária
    P_IINF = SIGMA_PIINF * A_SCP
    V_C, V_SW, A_SW = AREA_ACO_TRANSVERSAL_MODELO_I(0, P_IINF, V_SD, F_CTKINFJ[5], B_W, D, 'CP', W_INF, A_C, E_P, M_SD, F_CTMJ[5], 500E3)
    """
    print('Resistência do concreto ao cisalhamento')
    print('V_C:          ',   '%+10.5e' % V_C, 'kN')
    print('\n')
    print('Carga de cisalhamento absorvida pelo estribo')
    print('V_SW:         ',   '%+10.5e' % V_SW, 'kN')
    print('\n')
    print('Armadura para o cisalhamento')
    print('A_SW:         ',   '%+10.5e' % A_SW, 'm²/m')
    print('\n')
    """
    # Determinação das propriedades no Estádio I
    ALPHA_MOD = E_SCP / E_CSJ[5]
    A_CI, X_I, I_I = PROP_GEOMETRICA_ESTADIO_I(H, B_FS, B_W, H_FS, A_SCP, ALPHA_MOD, D)
    M_R = MOMENTO_RESISTENTE('I', F_CTMJ[5], H, X_I, I_I, P_IINF, A_CI, W_INF, E_P)
    # Determinação das propriedades no Estádio II
    X_II, I_II = PROP_GEOMETRICA_ESTADIO_II(H, B_FS, B_W, H_FS, A_SCP, 0, ALPHA_MOD, D, 0)
    """
    print("Verificação da flecha")
    print("Propriedades no Estádio I")
    print('Área da seção (Ac)               = ', A_CI, 'm²')
    print('Linha Neutra (xi)                = ', X_I, 'm')
    print('Inércia Estádio I (Ii)           = ', I_I, 'm^4')
    print('Momento resistente (Mr serviço)  = ', M_R, 'kN.m')
    print('\n')
    print("Propriedades no Estádio II")
    print('Linha Neutra Estádio II (xii)    = ', X_II, 'm')
    print('Inércia Estádio II (Iii)         = ', I_II, 'm^4')
    print('\n')
    """
    # Momento atuante
    M_SER = (M_MVG1K + M_MVG2K) + (M_MVQ1K) * PSI_2
    """
    print("Verificação do Estádio")
    print('Momento atuante (Mat serviço)    = ', M_SER, 'kN.m')
    """
    # Inércia e rigidez equivalente da peça
    if M_SER > M_R:
        I_M = INERCIA_BRANSON(M_R, M_SER, I_I, I_II)
        I_CNOVO = I_M
        #print('Peça deverá ser verificada no Estádio II')
    else:
        I_CNOVO  = I_I
        #print('Peça deverá ser verificada no Estádio I')
    #print('\n')

    EI0 = E_CSJ[0] * I_CNOVO
    EI1 = E_CSJ[5] * I_CNOVO
    """
    print("Avaliação da inércia equivalente")
    print("Inércia equivalente               =", I_CNOVO, 'm^4')
    print("Rigidez equivalente na protensão  =", EI0, 'kN.m²')
    print("Rigidez equivalente no serviço    =", EI1, 'kN.m²')
    """

    # Cálculo das flechas totais
    # Flecha na fabircação
    M_PI = P_I1 * E_P
    A_PII = -(M_PI * (L ** 2)) / (8 * EI0)
    A_PI = (PHI_GP + 1) * A_PII
    A_G11 = (5 * G_1K * (L ** 4) / (384 * EI0))
    A_G1 = (PHI_GP + 1) * A_G11
    A_FABRICA = np.abs(A_PII + A_G11)
    # Flecha no serviço
    A_G2 = (5 * G_2K * (L ** 4) / (384 * EI1))
    A_G2 *= (PHI_G2 + 1)
    A_Q1 = (5 * (PSI_2 * Q_1K) * (L ** 4) / (384 * EI1))
    A_Q1 *= (PHI_Q1 + 1)
    A_GT = A_PI + A_G1 + A_G2 + A_Q1
    A_SERVICO = A_GT

    # Verificação flecha na fabricação
    G_6 = VERIFICA_FLECHA(A_FABRICA, L / 1000)
    G_7 = VERIFICA_FLECHA(A_SERVICO, L / 250)
    """
    print("Flecha total (Direta + Fluência)")
    print("Flecha total na fabricação      = ", A_FABRICA, 'm', '// Eq. Estado Limite:', G_6)
    print("Flecha total no serviço         = ", A_SERVICO, 'm', '// Eq. Estado Limite:', G_7)
    """
    return G, A_C, A_SCP


# ## A célula abaixo verifica:
# - Propriedades geométricas da seção considerada;
# - Propriedades do material nas idades das etapas construtivas;
# - Tensão inicial nos cabos.

# ## A célula abaixo verifica:
# Esta célula verifica as tensões e faz a deifnição de armadura longitudinal para uma determinada perda de protensão. Porém o cálculo se repete até que a perda seja estabilizada.
# - O comprimento de transferência da peça;
# - Esforços de flexão meio do vão e apoios (no comp. de transferência);
# - Cisalhamento nos apoios;
# - Determinação da armadura A_SCP longitudinal que respeite a equação de Estado Limite de Serviço (ELS-F) no tempo infinito;
# - Atualização das perdas.

# ## A célula abaixo verifica:
# - Verifica as tensões normais no ato da protensão considerando um Estado Limite Último (ELU).

# ## A célula abaixo verifica:
# - Determina a armadura longitudinal necessária A_SCPNEC para a peça.

# ## A célula abaixo verifica:
# - Determina a resistência da biela de compressão;
# - Determina a armadura transversal A_SW necessária para a peça.

# ## A célula abaixo verifica:
# - Propriedades no Estádios I e II;
# - Momento de fissuração;
# - Flecha.
