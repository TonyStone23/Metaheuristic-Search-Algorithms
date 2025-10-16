#====================================================================
# General Data and Model Functions
# GA, CSA, and GTO for weight optimization of various regression models.
#--------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
#====================================================================
# General Data and Model Functions
#--------------------------------------------------------------------
# Collect simulated blueberry data
def getData(targets, predictors):
    data = pd.read_csv("blueberryDataCleaned.csv")
    X = np.asarray(data[predictors])
    ys = np.asarray(data[targets])
    return X, ys
#--------------------------------------------------------------------
# Visualize the data
def plotXs(predictors, X):
    # Plot by predictor category?
    fig, axs = plt.subplots(3, 5, figsize=(10, 6))
    axs = axs.flatten()
    for pred in range(len(predictors)):
        x = X[:,pred]
        axs[pred].plot(range(len(x)),x, color = 'blue')
        axs[pred].set_title(f"{predictors[pred]}")
    plt.suptitle("Blueberry Data Set Predictors")
    axs[-2].axis('off')
    axs[-1].axis('off')
    plt.tight_layout()
    plt.show()
    return
def plotYs(targets, ys):
    fig, axs = plt.subplots(2, 2, figsize = (8, 6))
    axs = axs.flatten()
    for tar in range(len(targets)):
        y = ys[:,tar]
        axs[tar].plot(range(len(y)), y, color = 'blue')
        axs[tar].set_title(f"{targets[tar]}")
    plt.suptitle("Blueberry Data Set Targets")
    plt.tight_layout()
    plt.show()
    return
#--------------------------------------------------------------------
# Performance metrics
def MSE(true, pred):
    return np.mean((pred - true) ** 2)
def MAE(true, pred):
    return np.mean(np.abs(pred - true))
def VAF(true, pred):
    return (1 - (np.var(true.flatten() - pred.flatten()) / np.var(true.flatten()))) * 100
def SSE(true, pred):
    return np.sum((true-pred)**2)
def R2(true, pred):
    return 1-(np.sum((true.flatten() - pred.flatten())**2)/np.sum((true.flatten() - np.mean(true.flatten()))**2))
#--------------------------------------------------------------------
# Generic model creation 
def getModel(n, type):
    global K
    global p
    global J
    global ip
    global h1
    global os
    global knots
    if type in ["linear", "log-linear", "poly"]:
        an = [0 for _ in range(n)]
    if type == "exponential":
        an = [0 for _ in range(n*2+1)]
    if type == "basis":
        p = 5
        an = [0 for _ in range(n*p+1)]
    if type == "gauss":
        J = 3
        an = [0 for _ in range(J*n + n +2)]
    if type == "fourier":
        K = 7
        an = [0 for _ in range(2 * n * K + 1)]
    if type == "mars":
        knots = 3
        an = [0 for _ in range(knots * n)]
    if type == "ANN":
        ip = n
        h1 = 10
        os = 1
        an = ip * h1 + h1 + h1 * os + os
        an = [0 for _ in range(an)]
    return an
#--------------------------------------------------------------------
# Make a prediction 
def predict(model, X, type):
    if type == "linear":
        pred = model[0]
        for i in range(0, len(model)-1):
            pred += model[i+1] * X[:,i]
    if type == "log-linear":
        pred = model[0]
        for i in range(0, len(model)-1):
            pred += np.exp(model[i+1] * X[:,i])
    if type == "poly":
        pred = model[0]
        _, n = X.shape
        d = range(0, n)
        for i in range(0, len(model)-1):
            pred += model[i+1] * X[:,i]**(d[i//n])
    if type == "exponential":
        pred = model[0]
        k = (len(model)-1)//2
        for i in range(0, k):
            pred += model[i+1]*np.exp(X[:,i] * model[i+k+1])
    if type == "power":
        pred = model[0]
        k = (len(model)-1)//2
        for i in range(0, k):
            pred += model[i+1]*(X[:,i] ** model[i+k+1])
    if type == "basis":
        pred = model[0]
        k = (len(model)-1)//p
        for j in range(1, k+1):
            for i in range(1, p+1):
                pred += model[(j-1)*p+1] *(X[:,j-1]**i)
    if type == "gauss":
        n = (len(model)-1)//(J+1)
        pred = model[0]
        l = model[1]
        B = model[2:n+2]
        mu = model[n+2:]
        for i in range(n):
            pred += B[i] * np.exp(-1/(2*l) * np.sum([X[:,i]- mu[j+i*J] for j in range(J)], axis=0))
    #"""
    if type == "fourier":
        pred = model[0]
        n = (len(model)-1) // (2 * K)
        k = (len(model)-1) // (2 * n)
        b = 1
        for j in range(n):
            for i in range(1, k+1):
                pred += (model[b] * np.sin(np.pi * i * X[:,j])) + (model[b+1] * np.cos(np.pi * i * X[:,j]))
                b += 2
    #"""
    """
    if type == "fourier":
        pred = model[0]
        n = (len(model)-1) // (2 * K)
        k = (len(model)-1) // (2 * n)
        b = 1
        for j in range(n):
            for i in range(1, k+1):
                pred += model[b] * np.sin(np.pi * i * X[:,j]) if (i//2 == 0) else model[b+1] * np.cos(np.pi * i * X[:,j])
                b += 1
    """
    if type == "mars": # multivariate adaptive spline regression
        pred = model[0]
        for k in range(knots):
            pred += pred

    if type == "ANN":
        idx = 0
        W1 = np.array(model[idx:idx + ip * h1]).reshape(ip, h1)
        idx += ip * h1
        b1 = np.array(model[idx:idx + h1])
        idx += h1
        W2 = np.array(model[idx:idx + h1 * os]).reshape(h1, os)
        idx += h1 * os
        b2 = np.array(model[idx:idx + os])
        hidden = np.tanh(np.dot(X, W1) + b1)
        pred = np.dot(hidden, W2) + b2
    return pred
#--------------------------------------------------------------------
#====================================================================
# Metaheuristic Search Algorithms for weight Optimization
#--------------------------------------------------------------------
# Fitness function
def fitness(ind, X, y, type):
    pred = predict(ind, X, type)
    return -np.mean((pred - y) ** 2)
#--------------------------------------------------------------------
# Crow Search algorithm
def CSA(model, type, X, y, pop_size = 25, generations = 1500, fl = 2, ap = 0.1, plot = True):
    print("beginning CSA")
    population = np.random.randn(pop_size, len(model))
    best_fitness_list = []
    avg_fitness_list = []
    dim = len(model)

    memory = population.copy()  
    scores = np.array([fitness(ind, X, y, type) for ind in population])
    memory_scores = scores.copy()

    for gen in range(generations):
        new_population = population.copy()

        for i in range(pop_size):
            j = np.random.randint(pop_size)           # crow to follow
            if np.random.rand() > ap:                 # j is unaware
                step = fl * np.random.rand(dim) * (memory[j] - population[i])
                new_population[i] = population[i] + step
            else:                                     # j is aware → random spot
                new_population[i] = np.random.randn(dim)

        new_scores = np.array([fitness(ind, X, y, type) for ind in new_population])

        improve_mask = new_scores > scores
        population[improve_mask] = new_population[improve_mask]
        scores[improve_mask] = new_scores[improve_mask]

        better_memory = scores > memory_scores
        memory[better_memory] = population[better_memory]
        memory_scores[better_memory] = scores[better_memory]

        best_fitness_list.append(-np.max(scores))  # Best MSE
        avg_fitness_list.append(-np.mean(scores))  # Average MSE

        if gen % 1 == 0:
            print(f"Generation {gen}: Best Fitness = {-np.max(scores):.5f}")

    if plot is True:
        # Plot convergence curve
        print("Convergence curve of CSA")
        plt.figure(figsize=(8, 4))
        plt.plot(best_fitness_list, label='Best Fitness (MSE)')
        plt.plot(avg_fitness_list, label='Average Fitness (MSE)', linestyle='--')
        plt.xlabel("Generation")
        plt.ylabel("MSE")
        plt.title("Convergence Curve of CSA")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    best_idx = np.argmax([fitness(ind, X, y, type) for ind in population])
    return population[best_idx] , best_fitness_list
#--------------------------------------------------------------------
# Gorilla Troop Optimizer
def GTO(model, type, X, y, pop_size = 25, generations = 1500, p = 0.03, b = 3, w = .8, plot = True):
    print("beginning GTO")
    population = np.random.randn(pop_size, len(model))
    best_fitness_list = []
    avg_fitness_list = []
    
    memory = population.copy()
    scores = np.array([fitness(ind, X, y, type) for ind in population])
    memory_scores = scores.copy()

    ub, lb = 1000, -1000

    for gen in range(generations):

        new_population = population.copy()

        f = np.cos(2 * np.random.random())+1
        c = f * (1- gen/generations)
        l = c * np.random.uniform(-1, 1)

        for i in range(pop_size):
            #Exploitation phase
            r1 = np.random.random()
            if r1 < p:
                new_population[i] = np.random.uniform(lb, ub, len(model))
            elif r1 >= .5:
                new_population[i] = (np.random.random()-c) * population[i] + l * np.random.uniform(-c, c) * population[i]
            else:
                r2 = np.random.randint(0, pop_size)
                new_population[i] = population[i]-l*(l*(population[i]-population[r2])+np.random.random()*(population[i]-population[r2]))

        new_scores = np.array([fitness(ind, X, y, type) for ind in new_population])
        
        improve_mask = new_scores > scores
        population[improve_mask] = new_population[improve_mask]
        scores[improve_mask] = new_scores[improve_mask]

        better_memory = scores > memory_scores
        memory[better_memory] = population[better_memory]
        memory_scores[better_memory] = scores[better_memory]

        best_idx = scores.argmax()
        silverback = population[best_idx]  # Average MSE
        candidates = new_population
        new_population = population.copy()
        
        g = 2**l
        e = np.random.normal(lb, ub) if np.random.uniform(0, 1) > .5 else np.random.normal()
        M = ((np.abs((1/pop_size)*np.sum(population, axis = 0)))**g)**(1/g)
        r3 = np.random.uniform(0, 1)
        for j in range(pop_size):
            # Exploitation phase
            if c >= w:
                new_population[j] = l * M *(population[j]-silverback)+population[j]
            else:
                new_population[j] = silverback-(silverback * (2 * r3 -1) - population[j] * (2 * r3 -1)) * b * e
        
        new_scores = np.array([fitness(ind, X, y, type) for ind in new_population])
        
        improve_mask = new_scores > scores
        population[improve_mask] = new_population[improve_mask]
        scores[improve_mask] = new_scores[improve_mask]

        better_memory = scores > memory_scores
        memory[better_memory] = population[better_memory]
        memory_scores[better_memory] = scores[better_memory]

        best_idx = scores.argmax()
        silverback = population[best_idx]
        best_fitness_list.append(-np.max(scores))  # Best MSE
        avg_fitness_list.append(-np.mean(scores))
        
        if gen % 1 == 0:
            print(f"Generation {gen}: Best Fitness = {-np.max(scores):.5f}")

    if plot is True:
        # Plot convergence curve
        print("Convergence curve of GTO")
        plt.figure(figsize=(8, 4))
        plt.plot(best_fitness_list, label='Best Fitness (MSE)')
        plt.plot(avg_fitness_list, label='Average Fitness (MSE)', linestyle='--')
        plt.xlabel("Generation")
        plt.ylabel("MSE")
        plt.title("Convergence Curve of GTO")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    best_idx = np.argmax([fitness(ind, X, y, type) for ind in population])
    return population[best_idx] , best_fitness_list   
#--------------------------------------------------------------------
def GA(model, type, X, y, pop_size = 25, generations = 1500, mutation_rate = .05, plot = True):
    population = np.random.randn(pop_size, len(model))*20
    best_fitness_list = []
    avg_fitness_list = []
    for gen in range(generations):
        scores = np.array([fitness(ind, X, y, type) for ind in population])
        best_fitness_list.append(-np.max(scores))  # Best MSE
        avg_fitness_list.append(-np.mean(scores))  # Average MSE

        # Select top 20% as parents
        top_idx = scores.argsort()[-pop_size // 5:]
        parents = population[top_idx]

        # Crossover and mutation
        new_population = []
        for _ in range(pop_size):
            p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
            point = np.random.randint(len(model))
            child = np.concatenate([p1[:point], p2[point:]])
            mutation = np.random.rand(len(model)) < mutation_rate
            child[mutation] += np.random.randn(np.sum(mutation)) * .5
            new_population.append(child)

        population = np.array(new_population)

        if gen % 10 == 0:
            print(f"Generation {gen}: Best Fitness = {-np.max(scores):.5f}")
    if plot is True:
        # Plot convergence curve
        plt.figure(figsize=(8, 4))
        plt.plot(best_fitness_list, label='Best Fitness (MSE)')
        plt.plot(avg_fitness_list, label='Average Fitness (MSE)', linestyle='--')
        plt.xlabel("Generation")
        plt.ylabel("MSE")
        plt.title("Convergence Curve of GA")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Return best individual
    best_idx = np.argmax([fitness(ind, X, y, type) for ind in population])
    return population[best_idx], best_fitness_list
#====================================================================
# Experiment
def Experiment(targetNum, targets, predictors, type, generations, plot = True):
    X, ys = getData(targets, predictors)
    y = ys[:,targetNum]
    X_train, X_test, y_train, y_test = tts(X, y, test_size = .3, random_state=42)
    
    s = StandardScaler()
    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)
    
    model = getModel(n = len(X[0]), type=type)

    best, bestlist = CSA(model=model, type=type, X=X_train, y=y_train, generations=generations)
    train_pred = predict(best, X_train, type=type)
    test_pred = predict(best, X_test, type=type)
    print(f"Best model: {best}")
    print(f"TRAINING R2 {R2(y_train, train_pred)}")
    print(f"TESTING R2 {R2(y_test, test_pred)}")

    if plot is True:
        plt.figure(figsize = (8, 6))

        plt.subplot(2, 1, 1)
        plt.plot(range(len(y_train)), train_pred, color = 'blue', label = "Predicted", linestyle = '-')
        plt.plot(range(len(y_train)), y_train, color = 'red', label = "Target", linestyle = '-')
        plt.title("Training Case")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(range(len(y_test)), test_pred, color = 'blue', label = "Predicted", linestyle = '-')
        plt.plot(range(len(y_test)), y_test, color = 'red', label = "Target", linestyle = '-')
        plt.title("Testing Case")
        plt.legend()
        plt.grid(True)

        plt.suptitle(f"Prediction of Blueberry {targets[targetNum]}")
        plt.tight_layout()
        plt.show()
#--------------------------------------------------------------------
# Run an experiment
"""
All Predictors:
    'clonesize',                                            # Average bluberry bush size
    'honeybee','bumbles','andrena', 'osmia',                # Bee density
    'MaxUC','MinUC','AveUC','MaxLC','MinLC','AveLC',        # Temperature
    'RDs','ARDs'                                            # Rain Related
"""
predictors = [
    'clonesize',
    'honeybee',
    'bumbles',
    'andrena',
    'osmia',
    'MaxUC',
    'MinUC',
    'AveUC',
    'MaxLC',
    'MinLC',
    'AveLC',
    'RDs',
    'ARDs'
    ]    
"""
ALL Targets:
    'fruitset',
    'fruitmass',
    'seeds',
    'yield'
"""
targets = [
    'fruitset',
    'fruitmass',
    'seeds',
    'yield'
    ]

Experiment(3, targets, predictors, type = 'basis', generations= 1500)
# 3, fourier, K = 7
# 3, gauss, J = 1

"""
Deepen ANN

Try Fourier Basis

Investigate GA and GTO
"""

