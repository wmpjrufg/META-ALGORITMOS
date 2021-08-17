# $$\      $$\ $$$$$$$$\ $$$$$$$$\  $$$$$$\        $$$$$$$$\  $$$$$$\   $$$$$$\  $$\       $$$$$$$\   $$$$$$\  $$\   $$\ 
# $$$\    $$$ |$$  _____|\__$$  __|$$  __$$\       \__$$  __|$$  __$$\ $$  __$$\ $$ |      $$  __$$\ $$  __$$\ $$ |  $$ |
# $$$$\  $$$$ |$$ |         $$ |   $$ /  $$ |         $$ |   $$ /  $$ |$$ /  $$ |$$ |      $$ |  $$ |$$ /  $$ |\$$\ $$  |
# $$\$$\$$ $$ |$$$$$\       $$ |   $$$$$$$$ |         $$ |   $$ |  $$ |$$ |  $$ |$$ |      $$$$$$$\ |$$ |  $$ | \$$$$  / 
# $$ \$$$  $$ |$$  __|      $$ |   $$  __$$ |         $$ |   $$ |  $$ |$$ |  $$ |$$ |      $$  __$$\ $$ |  $$ | $$  $$<  
# $$ |\$  /$$ |$$ |         $$ |   $$ |  $$ |         $$ |   $$ |  $$ |$$ |  $$ |$$ |      $$ |  $$ |$$ |  $$ |$$  /\$$\ 
# $$ | \_/ $$ |$$$$$$$$\    $$ |   $$ |  $$ |         $$ |    $$$$$$  | $$$$$$  |$$$$$$$$\ $$$$$$$  | $$$$$$  |$$ /  $$ |
# \__|     \__|\________|   \__|   \__|  \__|         \__|    \______/  \______/ \________|\_______/  \______/ \__|  \__|

################################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR,                  ENG. CIVIL / PROF (UFCAT)
# JOÃO V. COELHO ESTRELA,                                     ENG. MINAS (UFCAT)
################################################################################

################################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIOTECA META DE ALGORITMOS DE OTIMIZAÇÃO DESENVOLVIDA PELO GRUPO DE PESQUI-
# SAS E ESTUDOS EM ENGENHARIA (GPEE)
################################################################################

################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import numpy as np
import time

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE
import META_TOOLBOX.META_COMMON_LIBRARY as META_CL
import META_TOOLBOX.META_SA_ALGORITHM_LIBRARY as META_SA
import META_TOOLBOX.META_FA_ALGORITHM_LIBRARY as META_FA

# ALGORITMO DE RECOZIMENTO SIMULADO PADRÃO
def SA_ALGORITHM_0001(OF_FUNCTION, SETUP):
    """ 
    Standard Simulated Annealing algorithm.
   
    Input:
    OF_FUNCTION        | External def user input this function in arguments                        | Py function
    SETUP              | Algorithm setup                                                           | Py dictionary
                       |   Dictionary tags                                                         | 
                       |   'N_REP'  = Number of repetitions                                        | Integer
                       |   'N_ITER' = Number of iterations                                         | Integer
                       |   'N_POP'  = Number of population                                         | Integer
                       |   'D'      = Problem dimension                                            | Integer
                       |   'X_L'    = Lower limit design variables                                 | Py list[D]
                       |   'X_U'    = Upper limit design variables                                 | Py list[D]
                       |   'SIGMA'  = Standard deviation the normal distribution in percentage     | Float
                       |   'ALPHA'  = Temperature reduction factor - linear decay                  | Float
                       |   'TEMP'   = Initial temperature or automatic temperature value that has  | Float
                       |              an 80% probability of accepting the movement of particles    | 
                       |   'STOP_CONTROL_TEMP' = Stop criteria about initial temperature try       | Float
                       |                         or automatic value = 1000                         | 
                       |   'NULL_DIC' = Empty variable for the user to use in the obj. function    | ?
        
    Output:
    RESULTS_REP        | All results of population movement by repetition                          | Py dictionary
                       |   Dictionary tags                                                         |
                       |     'X_POSITION'    = Design variables by iteration                       | Py Numpy array[N_ITER + 1 x D]
                       |     'OF'            = Obj function value by iteration                     | Py Numpy array[N_ITER + 1 x 1]
                       |     'FIT'           = Fitness value by iteration                          | Py Numpy array[N_ITER + 1 x 1]
                       |     'SA_PARAMETERS' = Temperature by iteration                            | Py Numpy array[N_ITER + 1 x 1]
                       |     'NEOF'          = Number of objective function evaluations            | Py Numpy array[N_ITER + 1 x 1]
                       |     'ID_PARTICLE'   = ID particle                                         | Integer
    BEST_REP           | Best population results by repetition                                     | Py dictionary
                       |   Dictionary tags                                                         |
                       |     'X_POSITION'    = Design variables by iteration                       | Py Numpy array[N_ITER + 1 x D]
                       |     'OF'            = Obj function value by iteration                     | Py Numpy array[N_ITER + 1 x 1]
                       |     'FIT'           = Fitness value by iteration                          | Py Numpy array[N_ITER + 1 x 1]
                       |     'SA_PARAMETERS' = Temperature by iteration                            | Py Numpy array[N_ITER + 1 x 1]
                       |     'NEOF'          = Number of objective function evaluations            | Py Numpy array[N_ITER + 1 x 1]
                       |     'ID_PARTICLE'   = ID best particle by iteration                       | Integer    
    AVERAGE_REP        | Average OF and FIT results by repetition                                  | Py dictionary
                       |   Dictionary tags                                                         |
                       |     'OF'            = Obj function value by iteration                     | Py Numpy array[N_ITER + 1 x 1]
                       |     'FIT'           = Fitness value by iteration                          | Py Numpy array[N_ITER + 1 x 1]
                       |     'NEOF'          = Number of objective function evaluations            | Py Numpy array[N_ITER + 1 x 1]    
    WORST_REP          | Worst OF and FIT results by repetition                                    | Py dictionary
                       |   Dictionary tags                                                         |
                       |     'X_POSITION'    = Design variables by iteration                       | Py Numpy array[N_ITER + 1 x D]
                       |     'OF'            = Obj function value by iteration                     | Py Numpy array[N_ITER + 1 x 1]
                       |     'FIT'           = Fitness value by iteration                          | Py Numpy array[N_ITER + 1 x 1]
                       |     'SA_PARAMETERS' = Temperature by iteration                            | Py Numpy array[N_ITER + 1 x 1]
                       |     'NEOF'          = Number of objective function evaluations            | Py Numpy array[N_ITER + 1 x 1]
                       |     'ID_PARTICLE'   = ID best particle by iteration                       | Integer 
    STATUS_PROCEDURE   | Process repetition ID - from lowest OF value to highest OF value          | Py list[N_REP]
    """ 
    # Setup config
    N_REP = SETUP['N_REP']
    N_ITER = SETUP['N_ITER']
    N_POP = SETUP['N_POP']
    D = SETUP['D']
    X_L = SETUP['X_L']
    X_U = SETUP['X_U']
    SIGMA = SETUP['SIGMA']
    ALPHA = SETUP['ALPHA']
    TEMP = SETUP['TEMP']
    STOP_CONTROL_TEMP = SETUP['STOP_CONTROL_TEMP']
    NULL_DIC = SETUP['NULL_DIC']
    # Creating variables in the repetitions procedure
    RESULTS_REP = []
    BEST_REP = []
    WORST_REP = []
    AVERAGE_REP = []
    if NULL_DIC == None:
        NULL_DIC = []
    else:
        pass 
    # Repetition looping
    INIT = time.time()
    for I_COUNT in range(N_REP):
        # Creating variables in the iterations procedure
        X = np.zeros((N_POP, D)); OF = np.zeros((N_POP, 1)); FIT = np.zeros((N_POP, 1))
        RESULTS_ITER = [{'X_POSITION': np.empty((N_ITER + 1, D)), 'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'SA_PARAMETERS': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1), 'ID_PARTICLE': J_COUNT} for J_COUNT in range(N_POP)]
        BEST_ITER = {'X_POSITION': np.empty((N_ITER + 1, D)), 'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'SA_PARAMETERS': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1), 'ID_PARTICLE': np.empty(N_ITER + 1)}
        AVERAGE_ITER = {'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1)}
        WORST_ITER = {'X_POSITION': np.empty((N_ITER + 1, D)), 'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1), 'ID_PARTICLE': np.empty(N_ITER + 1)}
        STOP = 0
        NEOF_COUNT = 0 
        # Initial population
        X = META_CL.INITIAL_POPULATION(N_POP, D, X, X_L, X_U)
        for K_COUNT in range(N_POP):
            OF[K_COUNT, 0] = OF_FUNCTION(X[K_COUNT, :], NULL_DIC)
            FIT[K_COUNT, 0] = META_CL.FIT_VALUE(OF[K_COUNT, 0])
            NEOF_COUNT += 1
        # Initial temperature
        TEMPERATURE = META_SA.START_TEMPERATURE(OF_FUNCTION, NULL_DIC, N_POP, D, X, X_L, X_U, OF, SIGMA, TEMP, STOP_CONTROL_TEMP)
        # Storage all values in RESULTS_ITER
        for H_COUNT, X_ALL, OF_ALL, FIT_ALL, in zip(RESULTS_ITER, X, OF, FIT):
            H_COUNT['X_POSITION'][0, :] = X_ALL
            H_COUNT['OF'][0] = OF_ALL
            H_COUNT['FIT'][0] = FIT_ALL
            H_COUNT['SA_PARAMETERS'][0] = TEMPERATURE
            H_COUNT['NEOF'][0] = NEOF_COUNT
        # Best, Average and Worst storage
        BEST_POSITION, WORST_POSITION, X_BEST, X_WORST, OF_BEST, OF_WORST, FIT_BEST, FIT_WORST, OF_AVERAGE, FIT_AVERAGE = META_CL.BEST_VALUES(X, OF, FIT, N_POP)
        BEST_ITER['ID_PARTICLE'][0] = BEST_POSITION
        WORST_ITER['ID_PARTICLE'][0] = WORST_POSITION
        BEST_ITER['X_POSITION'][0, :] = X_BEST
        WORST_ITER['X_POSITION'][0, :] = X_WORST
        BEST_ITER['OF'][0] = OF_BEST
        AVERAGE_ITER['OF'][0] = OF_AVERAGE
        WORST_ITER['OF'][0] = OF_WORST
        BEST_ITER['FIT'][0] = FIT_BEST
        AVERAGE_ITER['FIT'][0] = FIT_AVERAGE
        WORST_ITER['FIT'][0] = FIT_WORST
        BEST_ITER['SA_PARAMETERS'][0] = TEMPERATURE
        BEST_ITER['NEOF'][0] = NEOF_COUNT
        AVERAGE_ITER['NEOF'][0] = NEOF_COUNT
        WORST_ITER['NEOF'][0] = NEOF_COUNT
        # Iteration procedure
        for L_COUNT in range(N_ITER):
            # SA particle movement
            for M_COUNT in range(N_POP):
                X_TEMPORARYI, OF_TEMPORARYI, FIT_TEMPORARYI, NEOF = META_SA.SA_MOVEMENT(OF_FUNCTION, NULL_DIC, X[M_COUNT, :], X_L, X_U, D, SIGMA)
                # Energy
                DELTAE = OF_TEMPORARYI - OF[M_COUNT, 0]
                # Probability of acceptance of the movement
                if DELTAE < 0:
                    PROBABILITY_STATE = 1
                elif DELTAE >= 0:
                    PROBABILITY_STATE = np.exp(- DELTAE / TEMPERATURE)
                RANDON_NUMBER = np.random.random()
                # New design variables
                if RANDON_NUMBER < PROBABILITY_STATE:
                    X[M_COUNT, :] = X_TEMPORARYI
                    OF[M_COUNT, 0] = OF_TEMPORARYI
                    FIT[M_COUNT, 0] = FIT_TEMPORARYI
                # Update NEOF (Number of Objective Function Evaluations)
                NEOF_COUNT += NEOF
            # Storage all values in RESULTS_ITER
            for J_COUNT, X_ALL, OF_ALL, FIT_ALL  in zip(RESULTS_ITER, X, OF, FIT):
                J_COUNT['X_POSITION'][L_COUNT + 1, :] = X_ALL
                J_COUNT['OF'][L_COUNT + 1] = OF_ALL
                J_COUNT['FIT'][L_COUNT + 1] = FIT_ALL
                J_COUNT['SA_PARAMETERS'][L_COUNT + 1] = TEMPERATURE
                J_COUNT['NEOF'][L_COUNT + 1] = NEOF_COUNT
            # Best, Average and Worst storage
            BEST_POSITION, WORST_POSITION, X_BEST, X_WORST, OF_BEST, OF_WORST, FIT_BEST, FIT_WORST, OF_AVERAGE, FIT_AVERAGE = META_CL.BEST_VALUES(X, OF, FIT, N_POP)
            BEST_ITER['ID_PARTICLE'][L_COUNT + 1] = BEST_POSITION
            WORST_ITER['ID_PARTICLE'][L_COUNT + 1] = WORST_POSITION
            BEST_ITER['X_POSITION'][L_COUNT + 1, :] = X_BEST
            WORST_ITER['X_POSITION'][L_COUNT + 1, :] = X_WORST
            BEST_ITER['OF'][L_COUNT + 1] = OF_BEST
            AVERAGE_ITER['OF'][L_COUNT + 1] = OF_AVERAGE
            WORST_ITER['OF'][L_COUNT + 1] = OF_WORST
            BEST_ITER['FIT'][L_COUNT + 1] = FIT_BEST
            AVERAGE_ITER['FIT'][L_COUNT + 1] = FIT_AVERAGE
            WORST_ITER['FIT'][L_COUNT + 1] = FIT_WORST
            BEST_ITER['SA_PARAMETERS'][L_COUNT + 1] = TEMPERATURE
            BEST_ITER['NEOF'][L_COUNT + 1] = NEOF_COUNT
            AVERAGE_ITER['NEOF'][L_COUNT + 1] = NEOF_COUNT
            WORST_ITER['NEOF'][L_COUNT + 1] = NEOF_COUNT
            # Update temperature (linear annealing schedule)
            TEMPERATURE = TEMPERATURE * ALPHA
        RESULTS_REP.append(RESULTS_ITER)
        BEST_REP.append(BEST_ITER)
        AVERAGE_REP.append(AVERAGE_ITER)
        WORST_REP.append(WORST_ITER)
        time.sleep(0.1)
        META_CL.PROGRESSBAR(I_COUNT + 1, N_REP, prefix = 'Progress:', suffix = 'Complete', length = 50)
    END = time.time()
    print('Process Time: %.2f' % (END - INIT), 'Seconds', '\n', 'Seconds per repetition: %.2f' % ((END - INIT) / N_REP))
    STATUS_PROCEDURE = META_CL.SUMMARY_ANALYSIS(BEST_REP, N_REP, N_ITER)
    return RESULTS_REP, BEST_REP, AVERAGE_REP, WORST_REP, STATUS_PROCEDURE

# ALGORITMO DE COLÔNIA DE VAGALUME PADRÃO
def FA_ALGORITHM_0001(OF_FUNCTION, SETUP):
    """
    Firefly algorithm.
    
    Input:
    OF_FUNCTION        | External def user input this function in arguments                        | Py function
    SETUP              | Algorithm setup                                                           | Py dictionary
                       |   Dictionary tags                                                         | 
                       |   'N_REP'     = Number of repetitions                                     | Integer
                       |   'N_ITER'    = Number of iterations                                      | Integer
                       |   'N_POP'     = Number of population                                      | Integer
                       |   'D'         = Problem dimension                                         | Integer
                       |   'X_L'       = Lower limit design variables                              | Py list[D]
                       |   'X_U'       = Upper limit design variables                              | Py list[D]
                       |   'BETA_0'    = Attractiveness at r = 0                                   | Float
                       |   'GAMMA'     = Light absorption coefficient  1 / (X_U - X_L) ** M        | Py list[D]
                       |   'ALPHA_MIN' = Min. randomic factor                                      | Float
                       |   'ALPHA_MAX' = Max. randomic factor                                      | Float
                       |   'THETA'     = Decrease in alpha factor                                  | Float    
                       |   'NULL_DIC'  = Empty variable for the user to use in the obj. function   | ?
        
    Output:
    RESULTS_REP        | All results of population movement by repetition                          | Py dictionary
                       |   Dictionary tags                                                         |
                       |     'X_POSITION'    = Design variables by iteration                       | Py Numpy array[N_ITER + 1 x D]
                       |     'OF'            = Obj function value by iteration                     | Py Numpy array[N_ITER + 1 x 1]
                       |     'FIT'           = Fitness value by iteration                          | Py Numpy array[N_ITER + 1 x 1]
                       |     'FA_PARAMETERS' = Alpha parameter by iteration                        | Py Numpy array[N_ITER + 1 x 1]
                       |     'NEOF'          = Number of objective function evaluations            | Py Numpy array[N_ITER + 1 x 1]
                       |     'ID_PARTICLE'   = ID particle                                         | Integer
    BEST_REP           | Best population results by repetition                                     | Py dictionary
                       |   Dictionary tags                                                         |
                       |     'X_POSITION'    = Design variables by iteration                       | Py Numpy array[N_ITER + 1 x D]
                       |     'OF'            = Obj function value by iteration                     | Py Numpy array[N_ITER + 1 x 1]
                       |     'FIT'           = Fitness value by iteration                          | Py Numpy array[N_ITER + 1 x 1]
                       |     'FA_PARAMETERS' = Alpha parameter by iteration                        | Py Numpy array[N_ITER + 1 x 1]
                       |     'NEOF'          = Number of objective function evaluations            | Py Numpy array[N_ITER + 1 x 1]
                       |     'ID_PARTICLE'   = ID best particle by iteration                       | Integer    
    AVERAGE_REP        | Average OF and FIT results by repetition                                  | Py dictionary
                       |   Dictionary tags                                                         |
                       |     'OF'            = Obj function value by iteration                     | Py Numpy array[N_ITER + 1 x 1]
                       |     'FIT'           = Fitness value by iteration                          | Py Numpy array[N_ITER + 1 x 1]
                       |     'NEOF'          = Number of objective function evaluations            | Py Numpy array[N_ITER + 1 x 1]    
    WORST_REP          | Worst OF and FIT results by repetition                                    | Py dictionary
                       |   Dictionary tags                                                         |
                       |     'X_POSITION'    = Design variables by iteration                       | Py Numpy array[N_ITER + 1 x D]
                       |     'OF'            = Obj function value by iteration                     | Py Numpy array[N_ITER + 1 x 1]
                       |     'FIT'           = Fitness value by iteration                          | Py Numpy array[N_ITER + 1 x 1]
                       |     'FA_PARAMETERS' = Alpha parameter by iteration                        | Py Numpy array[N_ITER + 1 x 1]
                       |     'NEOF'          = Number of objective function evaluations            | Py Numpy array[N_ITER + 1 x 1]
                       |     'ID_PARTICLE'   = ID best particle by iteration                       | Integer 
    STATUS_PROCEDURE   | Process repetition ID - from lowest OF value to highest OF value          | Py list[N_REP]
    """ 
    # Setup config
    N_REP = SETUP['N_REP']
    N_ITER = SETUP['N_ITER']
    N_POP = SETUP['N_POP']
    D = SETUP['D']
    X_L = SETUP['X_L']
    X_U = SETUP['X_U']
    BETA_0 = SETUP['BETA_0']
    GAMMA = SETUP['GAMMA']
    ALPHA_MIN = SETUP['ALPHA_MIN']
    ALPHA_MAX = SETUP['ALPHA_MAX']
    THETA = SETUP['THETA']
    NULL_DIC = SETUP['NULL_DIC']
    # Creating variables in the repetitions procedure
    RESULTS_REP = []
    BEST_REP = []
    WORST_REP = []
    AVERAGE_REP = []
    if NULL_DIC == None:
        NULL_DIC = []
    else:
        pass 
    # Repetition looping    
    INIT = time.time()
    for I_COUNT in range(N_REP):
        # Creating variables in the iterations procedure
        X = np.zeros((N_POP, D)); OF = np.zeros((N_POP, 1)); FIT = np.zeros((N_POP, 1))
        RESULTS_ITER = [{'X_POSITION': np.empty((N_ITER + 1, D)), 'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'FA_PARAMETERS': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1), 'ID_PARTICLE': J_COUNT} for J_COUNT in range(N_POP)]
        BEST_ITER = {'X_POSITION': np.empty((N_ITER + 1, D)), 'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'FA_PARAMETERS': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1), 'ID_PARTICLE': np.empty(N_ITER + 1)}
        AVERAGE_ITER = {'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1)}
        WORST_ITER = {'X_POSITION': np.empty((N_ITER + 1, D)), 'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1), 'ID_PARTICLE': np.empty(N_ITER + 1)}
        # Initial population
        NEOF_COUNT = 0
        X = META_CL.INITIAL_POPULATION(N_POP, D, X, X_L, X_U)
        ALPHA = ALPHA_MAX
        for K_COUNT in range(N_POP):
            OF[K_COUNT, 0] = OF_FUNCTION(X[K_COUNT, :], NULL_DIC)
            FIT[K_COUNT, 0] = META_CL.FIT_VALUE(OF[K_COUNT, 0])
            NEOF_COUNT += 1
        # Storage all values in RESULTS_ITER
        for H_COUNT, X_ALL, OF_ALL, FIT_ALL, in zip(RESULTS_ITER, X, OF, FIT):
            H_COUNT['X_POSITION'][0, :] = X_ALL
            H_COUNT['OF'][0] = OF_ALL
            H_COUNT['FIT'][0] = FIT_ALL
            H_COUNT['FA_PARAMETERS'][0] = ALPHA
            H_COUNT['NEOF'][0] = NEOF_COUNT
        # Best, Average and Worst storage
        BEST_POSITION, WORST_POSITION, X_BEST, X_WORST, OF_BEST, OF_WORST, FIT_BEST, FIT_WORST, OF_AVERAGE, FIT_AVERAGE = META_CL.BEST_VALUES(X, OF, FIT, N_POP)
        BEST_ITER['ID_PARTICLE'][0] = BEST_POSITION
        WORST_ITER['ID_PARTICLE'][0] = WORST_POSITION
        BEST_ITER['X_POSITION'][0, :] = X_BEST
        WORST_ITER['X_POSITION'][0, :] = X_WORST
        BEST_ITER['OF'][0] = OF_BEST
        AVERAGE_ITER['OF'][0] = OF_AVERAGE
        WORST_ITER['OF'][0] = OF_WORST
        BEST_ITER['FIT'][0] = FIT_BEST
        AVERAGE_ITER['FIT'][0] = FIT_AVERAGE
        WORST_ITER['FIT'][0] = FIT_WORST
        BEST_ITER['FA_PARAMETERS'][0] = ALPHA
        BEST_ITER['NEOF'][0] = NEOF_COUNT
        WORST_ITER['NEOF'][0] = NEOF_COUNT
        AVERAGE_ITER['NEOF'][0] = NEOF_COUNT
        # Iteration procedure
        for L_COUNT in range(N_ITER):
            # Ordering firefly according to fitness
            X_TEMP = X.copy()
            OF_TEMP = OF.copy()
            FIT_TEMP = FIT.copy()
            SORT_POSITIONS = np.argsort(OF_TEMP.T)
            for M_COUNT in range(N_POP):
                AUX = SORT_POSITIONS[0, M_COUNT]
                X[M_COUNT, :] = X_TEMP[AUX, :]
                OF[M_COUNT, 0] = OF_TEMP[AUX, 0] 
                FIT[M_COUNT, 0] = FIT_TEMP[AUX, 0]
            # Particle movement 
            X_J = X.copy()
            FITJ = FIT.copy()
            for N_COUNT in range(N_POP):
                for O_COUNT in range(N_POP):
                    FIT_I = FIT[N_COUNT, 0]; FIT_J = FITJ[O_COUNT, 0]
                    if FIT_I < FIT_J:
                        BETA = META_FA.ATTRACTIVENESS_FIREFLY_PARAMETER(BETA_0, GAMMA, X[N_COUNT, :], X_J[O_COUNT, :], D)                            
                        X_TEMPORARYI, OF_TEMPORARYI, FIT_TEMPORARYI, NEOF = META_FA.FIREFLY_MOVEMENT(OF_FUNCTION, X[N_COUNT, :], X_J[O_COUNT, :], BETA, ALPHA, D, X_L, X_U, NULL_DIC)
                    else:
                        X_TEMPORARYI = X[N_COUNT, :]
                        OF_TEMPORARYI = OF[N_COUNT, 0]
                        FIT_TEMPORARYI = FIT[N_COUNT, 0]
                        NEOF = 0
                    # New design variables
                    X[N_COUNT, :] = X_TEMPORARYI
                    OF[N_COUNT, 0] = OF_TEMPORARYI
                    FIT[N_COUNT, 0] = FIT_TEMPORARYI
                    NEOF_COUNT += NEOF
            # Storage all values in RESULTS_ITER
            for J_COUNT, X_ALL, OF_ALL, FIT_ALL  in zip(RESULTS_ITER, X, OF, FIT):
                J_COUNT['X_POSITION'][L_COUNT + 1, :] = X_ALL
                J_COUNT['OF'][L_COUNT + 1] = OF_ALL
                J_COUNT['FIT'][L_COUNT + 1] = FIT_ALL
                J_COUNT['FA_PARAMETERS'][L_COUNT + 1] = ALPHA
                J_COUNT['NEOF'][L_COUNT + 1] = NEOF_COUNT
            # Best, Average and Worst storage
            BEST_POSITION, WORST_POSITION, X_BEST, X_WORST, OF_BEST, OF_WORST, FIT_BEST, FIT_WORST, OF_AVERAGE, FIT_AVERAGE = META_CL.BEST_VALUES(X, OF, FIT, N_POP)
            BEST_ITER['ID_PARTICLE'][L_COUNT + 1] = BEST_POSITION
            WORST_ITER['ID_PARTICLE'][L_COUNT + 1] = WORST_POSITION
            BEST_ITER['X_POSITION'][L_COUNT + 1, :] = X_BEST
            WORST_ITER['X_POSITION'][L_COUNT + 1, :] = X_WORST
            BEST_ITER['OF'][L_COUNT + 1] = OF_BEST
            AVERAGE_ITER['OF'][L_COUNT + 1] = OF_AVERAGE
            WORST_ITER['OF'][L_COUNT + 1] = OF_WORST
            BEST_ITER['FIT'][L_COUNT + 1] = FIT_BEST
            AVERAGE_ITER['FIT'][L_COUNT + 1] = FIT_AVERAGE
            WORST_ITER['FIT'][L_COUNT + 1] = FIT_WORST
            BEST_ITER['FA_PARAMETERS'][L_COUNT + 1] = ALPHA
            BEST_ITER['NEOF'][L_COUNT + 1] = NEOF_COUNT
            AVERAGE_ITER['NEOF'][L_COUNT + 1] = NEOF_COUNT
            WORST_ITER['NEOF'][L_COUNT + 1] = NEOF_COUNT
            # Update alpha factor
            # ALPHA = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * np.exp(- L_COUNT)
            ALPHA = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * THETA ** L_COUNT
        RESULTS_REP.append(RESULTS_ITER)
        BEST_REP.append(BEST_ITER)
        AVERAGE_REP.append(AVERAGE_ITER)
        WORST_REP.append(WORST_ITER)
        time.sleep(0.1)
        META_CL.PROGRESSBAR(I_COUNT + 1, N_REP, prefix = 'Progress:', suffix = 'Complete', length = 50)
    END = time.time()
    print('Process Time: %.2f' % (END - INIT), 'Seconds', '\n', 'Seconds per repetition: %.2f' % ((END - INIT) / N_REP))
    STATUS_PROCEDURE = META_CL.SUMMARY_ANALYSIS(BEST_REP, N_REP, N_ITER)
    return RESULTS_REP, BEST_REP, AVERAGE_REP, WORST_REP, STATUS_PROCEDURE

# ALGORITMO DE COLÔNIA DE VAGALUME COM GÊNERO
def FA_ALGORITHM_0002(OF_FUNCTION, N_REP, N_ITER, N_POPMALE, N_POPFEMALE, D, TYPE_BOOT, X_L, X_U, BETA_0, GAMMA, N_CHAOTICSEARCHS, ALPHA_CHAOTIC, NULL_DIC = None):
    """ 
    CONTINUOUS GENDER FIREFLY ALGORITHM

    INPUT:
    OF_FUNCTION        | External def user input this function in arguments                        | Py function
    N_REP              | Number of repetitions                                                     | Integer
    N_ITER             | Number of iterations                                                      | Integer
    N_POP              | Number of population                                                      | Integer
    D                  | Problem dimension                                                         | Integer
    X_L                | Lower limit design variables                                              | Py list[D]
    X_U                | Upper limit design variables                                              | Py list[D]
    BETA_0:  (FLOAT)
    GAMMA:  (FLOAT)
    N_CHAOTICSEARCHS:  (FLOAT)
    ALPHA_CHAOTIC:  (FLOAT)
    NULL_DIC: NULL DICTIONARY OR AUTOMATIC VALUE (DICTIONARY, MIXED)
        
    OUTPUT:
    RESULTS_REP: RESULTS MOVEMENT ALL POPULATION, (DICTIONARY, MIXED)
    BEST_REP: RESULTS MOVEMENT BEST POPULATION, (DICTIONARY, MIXED)
    MEAN_REP: RESULTS MOVEMENT MEAN FO AND FIT, (DICTIONARY, MIXED)
    WORST_REP: RESULTS MOVEMENT WORST POPULATION, (DICTIONARY, MIXED)
    """

    # START RESERVED SPACE FOR REPETITIONS
    RESULTS_REP = []
    BEST_REP = []
    WORST_REP = []
    AVERAGE_REP = []
    if NULL_DIC == None:
        NULL_DIC = []
    else:
        pass 
    # REPETITION LOOPING    
    for I_COUNT in range(N_REP):
        # START RESERVED SPACE FOR ITERATIONS
        X_MALE = np.zeros((N_POPMALE, D))
        X_MALETEMPORARY = np.zeros((N_POPMALE, D))
        OF_MALE = np.zeros((N_POPMALE, 1))
        OF_MALETEMPORARY = np.zeros((N_POPMALE, 1))
        FIT_MALE = np.zeros((N_POPMALE, 1))
        FIT_MALETEMPORARY = np.zeros((N_POPMALE, 1))
        Y_FEMALE = np.zeros((N_POPFEMALE, D))
        Y_FEMALETEMPORARY = np.zeros((N_POPFEMALE, D))
        OF_FEMALE = np.zeros((N_POPFEMALE, 1))
        OF_FEMALETEMPORARY = np.zeros((N_POPFEMALE, 1))
        FIT_FEMALE = np.zeros((N_POPFEMALE, 1))
        FIT_FEMALETEMPORARY = np.zeros((N_POPFEMALE, 1))
        RESULTS_ITER = [{'X_MALEPOSITION': np.empty((N_ITER + 1, D)), 'OF_MALE': np.empty(N_ITER + 1), 'FIT_MALE': np.empty(N_ITER + 1), 'Y_FEMALEPOSITION': np.empty((N_ITER + 1, D)), 'OF_FEMALE': np.empty(N_ITER + 1), 'FIT_FEMALE': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1), 'ID_PARTICLE': J_COUNT} for J_COUNT in range(N_POPMALE)]
        BEST_ITER = {'X_POSITION': np.empty((N_ITER + 1, D)), 'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'FA_PARAMETERS': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1), 'ID_PARTICLE': np.empty(N_ITER + 1)}
        AVERAGE_ITER = {'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1),}
        WORST_ITER = {'X_POSITION': np.empty((N_ITER + 1, D)), 'OF': np.empty(N_ITER + 1), 'FIT': np.empty(N_ITER + 1), 'NEOF': np.empty(N_ITER + 1), 'ID_PARTICLE': np.empty(N_ITER + 1)}
        # INITIAL POPULATION
        X_MALE = META_CL.INITIAL_POPULATION(N_POPMALE, D, X_MALE, X_L, X_U)
        Y_FEMALE = META_CL.INITIAL_POPULATION(N_POPFEMALE, D, Y_FEMALE, X_L, X_U)
        # EVALUATED OF AND FIT
        for K_COUNT in range(N_POPMALE):
            OF_MALE[K_COUNT, 0] = OF_FUNCTION(X_MALE[K_COUNT, :], NULL_DIC)
            FIT_MALE[K_COUNT, 0] = META_CL.FIT_VALUE(OF_MALE[K_COUNT, 0])
            OF_FEMALE[K_COUNT, 0] = OF_FUNCTION(Y_FEMALE[K_COUNT, :], NULL_DIC)
            FIT_FEMALE[K_COUNT, 0] = META_CL.FIT_VALUE(OF_FEMALE[K_COUNT, 0])
        # STORAGE IN RESULTS_ITER
        for L_COUNT, X_MALEI, OF_MALEI, FIT_MALEI, Y_FEMALEK, OF_FEMALEK, FIT_FEMALEK  in zip(RESULTS_ITER, X_MALE, OF_MALE, FIT_MALE, Y_FEMALE, OF_FEMALE, FIT_FEMALE):
            L_COUNT['X_MALEPOSITION'][0, :] = X_MALEI
            L_COUNT['OF_MALE'][0] = OF_MALEI
            L_COUNT['FIT_MALE'][0] = FIT_MALEI
            L_COUNT['Y_FEMALEPOSITION'][0, :] = Y_FEMALEK
            L_COUNT['OF_FEMALE'][0] = OF_FEMALEK
            L_COUNT['FIT_FEMALE'][0] = FIT_FEMALEK
        # BEST MEAN, AND WORST STORAGE
        X = np.concatenate((X_MALE, Y_FEMALE), axis = 0)
        OF = np.concatenate((OF_MALE, OF_FEMALE), axis = 0)
        FIT = np.concatenate((FIT_MALE, FIT_FEMALE), axis = 0)
        N_POP = N_POPFEMALE + N_POPMALE
        BEST_POSITION, WORST_POSITION, X_BEST, X_WORST, OF_BEST, OF_WORST, FIT_BEST, FIT_WORST, OF_AVERAGE, FIT_AVERAGE = META_CL.BEST_VALUES(X, OF, FIT, N_POP)
        BEST_ITER['ID_PARTICLE'][0] = BEST_POSITION
        WORST_ITER['ID_PARTICLE'][0] = WORST_POSITION
        BEST_ITER['X_POSITION'][0, :] = X_BEST
        WORST_ITER['X_POSITION'][0, :] = X_WORST
        BEST_ITER['OF'][0] = OF_BEST
        AVERAGE_ITER['OF'][0] = OF_AVERAGE
        WORST_ITER['OF'][0] = OF_WORST
        BEST_ITER['FIT'][0] = FIT_BEST
        AVERAGE_ITER['FIT'][0] = FIT_AVERAGE
        WORST_ITER['FIT'][0] = FIT_WORST
        # ITERATION PROCEDURE
        for M_COUNT in range(N_ITER):
            # BEST MALE
            _, _, X_MALEBEST, _, _, _, FIT_MALEBEST, _, _, _= META_CL.BEST_VALUES(X_MALE, OF_MALE, FIT_MALE, N_POPMALE) 
            # GFA MOVEMENT PARTICLE
            for N_COUNT in range(N_POPMALE):
                # SELECTION RANDOM FEMALES
                FEMALES = np.random.randint(0, (N_POPFEMALE - 1), size = 2)
                K = FEMALES[0]
                J = FEMALES[1]
                # MALE MOVEMENT
                X_MALETEMPORARY[N_COUNT, :], OF_MALETEMPORARY[N_COUNT, 0], FIT_MALETEMPORARY[N_COUNT, 0] = META_FA.MALE_FIREFLY_MOVEMENT(OF_FUNCTION, X_MALE[N_COUNT, :], FIT_MALE[N_COUNT, 0], Y_FEMALE[K, :], FIT_FEMALE[K, 0], Y_FEMALE[J, :], FIT_FEMALE[J, 0], BETA_0, GAMMA, D, X_L, X_U, NULL_DIC)
                # FEMALE MOVEMENT
                Y_FEMALETEMPORARY[N_COUNT, :], OF_FEMALETEMPORARY[N_COUNT, 0], FIT_FEMALETEMPORARY[N_COUNT, 0] = META_FA.FEMALE_FIREFLY_MOVEMENT(OF_FUNCTION, Y_FEMALE[N_COUNT, :], X_MALEBEST, FIT_MALEBEST, BETA_0, GAMMA, D, X_L, X_U, NULL_DIC)
            # GLOBAL BEST SELECTION AND STORAGE    
            X = np.concatenate((X_MALETEMPORARY, Y_FEMALETEMPORARY), axis = 0)
            OF = np.concatenate((OF_MALETEMPORARY, OF_FEMALETEMPORARY), axis = 0)
            FIT = np.concatenate((FIT_MALETEMPORARY, FIT_FEMALETEMPORARY), axis = 0)
            BEST_POSITION1, WORST_POSITION1, X_BEST1, X_WORST1, OF_BEST1, OF_WORST1, FIT_BEST1, FIT_WORST1, _, _ = META_CL.BEST_VALUES(X, OF, FIT, N_POP)
            if FIT_BEST1 > FIT_BEST:
                BEST_POSITION =  BEST_POSITION1
                WORST_POSITION = WORST_POSITION1
                X_BEST = X_BEST1
                X_WORST = X_WORST1
                OF_BEST = OF_BEST1
                OF_WORST = OF_WORST1
                FIT_BEST = FIT_BEST1
                FIT_WORST = FIT_WORST1
            X_BEST2, OF_BEST2, FIT_BEST2 = META_FA.CHAOTIC_SEARCH(OF_FUNCTION, M_COUNT, X_BEST, OF_BEST, FIT_BEST, N_CHAOTICSEARCHS, ALPHA_CHAOTIC, D, N_ITER, X_L, X_U, NULL_DIC)
            if FIT_BEST2 > FIT_BEST:
                X_BEST = X_BEST2
                OF_BEST = OF_BEST2
                FIT_BEST = FIT_BEST2
            if FIT_BEST2 > FIT_BEST1:  
                if BEST_POSITION < (N_POPMALE - 1):
                    X_MALETEMPORARY[BEST_POSITION, :] = X_BEST2
                    OF_MALETEMPORARY[BEST_POSITION, 0] = OF_BEST2
                    FIT_MALETEMPORARY[BEST_POSITION, 0] = FIT_BEST2
                else:
                    Y_FEMALETEMPORARY[BEST_POSITION - N_POPFEMALE, :] = X_BEST2
                    OF_FEMALETEMPORARY[BEST_POSITION - N_POPFEMALE, 0] = OF_BEST2
                    FIT_FEMALETEMPORARY[BEST_POSITION - N_POPFEMALE, 0] = FIT_BEST2
            # STORAGE IN RESULTS_ITER
            X_MALE = X_MALETEMPORARY
            OF_MALE = OF_MALETEMPORARY
            FIT_MALE = FIT_MALETEMPORARY
            Y_FEMALE = Y_FEMALETEMPORARY
            OF_FEMALE = OF_FEMALETEMPORARY
            FIT_FEMALE = FIT_FEMALETEMPORARY
            for O_COUNT, X_MALEI, OF_MALEI, FIT_MALEI, Y_FEMALEK, OF_FEMALEK, FIT_FEMALEK  in zip(RESULTS_ITER, X_MALE, OF_MALE, FIT_MALE, Y_FEMALE, OF_FEMALE, FIT_FEMALE):
                O_COUNT['X_MALEPOSITION'][M_COUNT + 1, :] = X_MALEI
                O_COUNT['OF_MALE'][M_COUNT + 1] = OF_MALEI
                O_COUNT['FIT_MALE'][M_COUNT + 1] = FIT_MALEI
                O_COUNT['Y_FEMALEPOSITION'][M_COUNT + 1, :] = Y_FEMALEK
                O_COUNT['OF_FEMALE'][M_COUNT + 1] = OF_FEMALEK
                O_COUNT['FIT_FEMALE'][M_COUNT + 1] = FIT_FEMALEK
            X = np.concatenate((X_MALE, Y_FEMALE), axis = 0)
            OF = np.concatenate((OF_MALE, OF_FEMALE), axis = 0)
            FIT = np.concatenate((FIT_MALE, FIT_FEMALE), axis = 0)
            BEST_ITER['ID_PARTICLE'][M_COUNT + 1] = BEST_POSITION
            WORST_ITER['ID_PARTICLE'][M_COUNT + 1] = WORST_POSITION
            BEST_ITER['X_POSITION'][M_COUNT + 1, :] = X_BEST
            WORST_ITER['X_POSITION'][M_COUNT + 1, :] = X_WORST
            BEST_ITER['OF'][M_COUNT + 1] = OF_BEST
            AVERAGE_ITER['OF'][M_COUNT + 1] = np.mean(OF)
            WORST_ITER['OF'][M_COUNT + 1] = OF_WORST
            BEST_ITER['FIT'][M_COUNT + 1] = FIT_BEST
            AVERAGE_ITER['FIT'][M_COUNT + 1] = np.mean(FIT)
            WORST_ITER['FIT'][M_COUNT + 1] = FIT_WORST
            RESULTS_REP.append(RESULTS_ITER)
            BEST_REP.append(BEST_ITER)
            AVERAGE_REP.append(AVERAGE_ITER)
            WORST_REP.append(WORST_ITER)
    return RESULTS_REP, BEST_REP, AVERAGE_REP, WORST_REP

#   /$$$$$$  /$$$$$$$  /$$$$$$$$ /$$$$$$$$       /$$$$$$$$ /$$$$$$$$  /$$$$$$  /$$   /$$ /$$   /$$  /$$$$$$  /$$        /$$$$$$   /$$$$$$  /$$$$$$ /$$$$$$$$  /$$$$$$ 
#  /$$__  $$| $$__  $$| $$_____/| $$_____/      |__  $$__/| $$_____/ /$$__  $$| $$  | $$| $$$ | $$ /$$__  $$| $$       /$$__  $$ /$$__  $$|_  $$_/| $$_____/ /$$__  $$
# | $$  \__/| $$  \ $$| $$      | $$               | $$   | $$      | $$  \__/| $$  | $$| $$$$| $$| $$  \ $$| $$      | $$  \ $$| $$  \__/  | $$  | $$      | $$  \__/
# | $$ /$$$$| $$$$$$$/| $$$$$   | $$$$$            | $$   | $$$$$   | $$      | $$$$$$$$| $$ $$ $$| $$  | $$| $$      | $$  | $$| $$ /$$$$  | $$  | $$$$$   |  $$$$$$ 
# | $$|_  $$| $$____/ | $$__/   | $$__/            | $$   | $$__/   | $$      | $$__  $$| $$  $$$$| $$  | $$| $$      | $$  | $$| $$|_  $$  | $$  | $$__/    \____  $$
# | $$  \ $$| $$      | $$      | $$               | $$   | $$      | $$    $$| $$  | $$| $$\  $$$| $$  | $$| $$      | $$  | $$| $$  \ $$  | $$  | $$       /$$  \ $$
# |  $$$$$$/| $$      | $$$$$$$$| $$$$$$$$         | $$   | $$$$$$$$|  $$$$$$/| $$  | $$| $$ \  $$|  $$$$$$/| $$$$$$$$|  $$$$$$/|  $$$$$$/ /$$$$$$| $$$$$$$$|  $$$$$$/
#  \______/ |__/      |________/|________/         |__/   |________/ \______/ |__/  |__/|__/  \__/ \______/ |________/ \______/  \______/ |______/|________/ \______/ 