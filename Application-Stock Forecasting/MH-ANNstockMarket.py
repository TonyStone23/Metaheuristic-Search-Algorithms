#====================================================================================
# Stock Price prediction with MH-ANN
#====================================================================================
import pandas as pd
import random as r
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# Set seed for reproducability

seed = 23
r.seed(seed)
np.random.seed(seed)

def getStock(num):
    if num == 1: #DJI
        training = pd.read_csv("Stock Time Series/Data/DJA_training.csv")
        validation = pd.read_csv("Stock Time Series/Data/DJA_validation.csv")
    if num == 2: #S&P
        training = pd.read_csv("Stock Time Series/Data/^SPX_prior_five_years.csv")
        validation = pd.read_csv("Stock Time Series/Data/^SPX_last_year.csv")

    return training, validation

training, validation = getStock(2)

y = training['Close']
y = y[1:].to_numpy()
X = training[['Open', 'Close', 'High', 'Low']]
X = X[:-1].to_numpy()

s = StandardScaler()
s.fit_transform(X)

# Structure of the ANN
input_size = X.shape[1]
hidden_size1 = 8
hidden_size2 = 12
output_size = 1

weight_size = input_size * hidden_size1 + hidden_size1 + hidden_size1 * hidden_size2 + hidden_size2 + hidden_size2 * output_size + output_size
# Performance metrics
def MSE(true, pred):
    return np.mean((pred - true) ** 2)
def MAE(true, pred):
    return np.mean(np.abs(pred - true))
def VAF(true, pred):
    return (1 - (np.var(true - pred) / np.var(true))) * 100
def SSE(true, pred):
    return np.sum((true-pred)**2)
def R2(true, pred):
    return 1-(np.sum((true - pred)**2)/np.sum((true - np.mean(true))**2))

# Neural network forward pass
def forward_pass(X, weights):
    """
    W1, b1, W2, b2, W3, b3 = unpack_weights(weights)
    hidden1 = np.tanh(np.dot(X, W1) + b1)
    hidden2 = np.tanh(np.dot(hidden1, W2) + b2)
    output = np.dot(hidden2, W3) + b3"""
    
    
    W1, b1, W2, b2, W3, b3 = unpack_weights(weights)
    hidden1 = np.maximum(0, np.dot(X, W1) + b1)
    hidden2 = np.maximum(0, np.dot(hidden1, W2) + b2)
    output = np.dot(hidden2, W3) + b3
    
    return output.flatten()

# Unpack flat weight vector into layer weights and biases
def unpack_weights(weights):

    idx = 0 
    W1 = weights[idx:idx + input_size*hidden_size1].reshape(input_size, hidden_size1)
    idx += input_size*hidden_size1
    b1 = weights[idx:idx + hidden_size1]
    idx += hidden_size1
    
    W2 = weights[idx:idx + hidden_size1*hidden_size2].reshape(hidden_size1, hidden_size2)
    idx += hidden_size1*hidden_size2
    b2 = weights[idx:idx + hidden_size2]
    idx += hidden_size2
    
    W3 = weights[idx:idx + hidden_size2*output_size].reshape(hidden_size2, output_size)
    idx += hidden_size2*output_size
    b3 = weights[idx:idx + output_size]
    
    return W1, b1, W2, b2, W3, b3

def pack_weights(W1, b1, W2, b2, W3, b3):
    weights = np.concatenate([
        W1.flatten(),
        b1.flatten(),
        W2.flatten(),
        b2.flatten(),
        W3.flatten(),
        b3.flatten()
    ])
    return weights
# Fitness function (maximize negative MSE)
def fitness(X, y, weights):
    y_pred = forward_pass(X, weights)
    return -np.mean((y - y_pred) ** 2)

# Autoregression algorithm
def AR(data, lags = 1):
    print("Starting AR")
    temp = pd.DataFrame(data, columns=["Close"])
    
    for k in range(1, lags + 1):
        temp[f"P-{k}"] = temp['Close'].shift(k)

    temp = temp.dropna().reset_index(drop=True)

    y = temp['Close'].to_numpy()
    X_ar = temp[[f"P-{k}" for k in range(1, lags + 1)]].to_numpy()

    X_design = np.column_stack([np.ones(len(X_ar)), X_ar])

    coeffs, *_ = np.linalg.lstsq(X_design, y, rcond = None)

    pred = X_design @ coeffs

    return pred, coeffs

def validate_AR(data, coeffs):
    intercept = coeffs[0]
    phi = coeffs[1:]

    temp = pd.DataFrame(data, columns=["Close"])
    
    for k in range(1, lags + 1):
        temp[f"P-{k}"] = temp['Close'].shift(k)

    temp = temp.dropna().reset_index(drop=True)

    y = temp['Close'].to_numpy()
    X_ar = temp[[f"P-{k}" for k in range(1, lags + 1)]].to_numpy()

    pred = intercept + np.dot(X_ar, phi)

    return pred

# Genetic Algorithm with convergence tracking
def GA(data, y, pop_size=150, generations=1500, mutation_rate=0.025, plot = True):
    print("   Starting GA")
    population = np.random.randn(pop_size, weight_size) * np.sqrt(2.0 / np.mean((input_size + hidden_size1 + hidden_size2)))
    best_fitness_list = []
    avg_fitness_list = []

    for gen in range(generations):
        scores = np.array([fitness(data, y, ind) for ind in population])
        best_fitness_list.append(-np.max(scores))  # Best MSE
        avg_fitness_list.append(-np.mean(scores))  # Average MSE

        # Select top 20% as parents
        top_idx = scores.argsort()[-pop_size // 5:]
        parents = population[top_idx]

        # Crossover and mutation
        new_population = []
        for _ in range(pop_size):
            p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
            point1 = np.random.randint(0, weight_size//2)
            point2 = np.random.randint(weight_size//2, weight_size)
            child = np.concatenate([p1[:point1], p2[point1:point2], p1[point2:]])
            mutation = np.random.rand(weight_size) < mutation_rate
            child[mutation] += np.random.randn(np.sum(mutation)) * 5
            new_population.append(child)

        population = np.array(new_population)

        if gen % 1 == 0:
            print(f"Generation {gen}: Best Fitness = {-np.max(scores):.5f}")

    if plot is True:
        # Plot convergence curve
        print("Convergence curve of GA")
        plt.figure(figsize=(8, 4))
        plt.plot(best_fitness_list, label='Best Fitness (MSE)')
        plt.plot(avg_fitness_list, label='Average Fitness (MSE)', linestyle='--')
        plt.xlabel("Generation")
        plt.ylabel("MSE")
        #plt.title("Convergence Curve of GA")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Return best individual
    best_idx = np.argmax([fitness(data, y, ind) for ind in population])
    return population[best_idx], best_fitness_list

# Crow Search Algorithm with convergence tracking
def CSA(data, y, pop_size = 150, generations = 1500, fl = 1.25, ap = 0.25, plot = True):
    print("   Starting CSA")
    population = np.random.randn(pop_size, weight_size) * np.sqrt(2.0 / np.mean((input_size + hidden_size1 + hidden_size2)))
    best_fitness_list = []
    avg_fitness_list = []
    dim = weight_size

    memory = population.copy()  
    scores = np.array([fitness(data, y, ind) for ind in population])
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

        new_scores = np.array([fitness(data, y, ind) for ind in new_population])

        improve_mask = new_scores > scores
        population[improve_mask] = new_population[improve_mask]
        scores[improve_mask]     = new_scores[improve_mask]

        better_memory = scores > memory_scores
        memory[better_memory]        = population[better_memory]
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
        #plt.title("Convergence Curve of CSA")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    best_idx = np.argmax([fitness(data, y, ind) for ind in population])
    return population[best_idx] , best_fitness_list

def BP(data, y, epochs, learning_rate = .00000001):
    
    print("   Starting BP")
    best_loss_list = []
    weights = np.random.randn(1, weight_size) * np.sqrt(2.0 / np.mean((input_size + hidden_size1 + hidden_size2)))
    weights = weights.flatten()
    
    for epoch in range(epochs):
        W1, b1, W2, b2, W3, b3 = unpack_weights(weights)
        z1 = np.dot(data, W1) + b1
        a1 = np.maximum(0, np.dot(data, W1) + b1)
        z2 = np.dot(a1, W2) + b2
        a2 = np.maximum(0, z2) 
        z3 = np.dot(a2, W3) + b3
        output = z3
        
        # Compute loss (Mean Squared Error)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if output.ndim == 1:
            output = output.reshape(-1, 1)
            
        loss = np.mean((output - y) ** 2)
        best_loss_list.append(loss)
        
        # Backward pass
        m = X.shape[0]  # Number of samples
        
        # Output layer gradients
        dz3 = (output - y) / m
        dW3 = np.dot(a2.T, dz3)
        db3 = np.sum(dz3, axis=0)
        
        # Hidden layer 2 gradients
        da2 = np.dot(dz3, W3.T)
        dz2 = da2 * (z2 > 0)  # ReLU derivative
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        
        # Hidden layer 1 gradients
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * (z1 > 0)  # ReLU derivative
        dW1 = np.dot(data.T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        
        # Pack weights back into flat vector
        weights = pack_weights(W1, b1, W2, b2, W3, b3)
    
    return weights, best_loss_list
# Run Algorithms and get best weights

# Standard Parameters
runs = 5
pop_size = 200
generations = 100

#AR metrics
AR_mses = []
AR_maes = []
AR_vafs = []
AR_sses = []
AR_r2s = []
AR_best_mse = np.inf
AR_best_weights = None
AR_best_convergence = None
AR_convergences = []
lags = 3
#GA metrics
GA_mses = []
GA_maes = []
GA_vafs = []
GA_sses = []
GA_r2s = []
GA_best_mse = np.inf
GA_best_weights = None
GA_best_convergence = None
GA_convergences = []
#CSA metrics
CSA_mses = []
CSA_maes = []
CSA_vafs = []
CSA_sses = []
CSA_r2s = []
CSA_best_mse = np.inf
CSA_best_weights = None
CSA_best_convergence = None
CSA_convergences = []
#BP metrics
BP_mses = []
BP_maes = []
BP_vafs = []
BP_sses = []
BP_r2s = []
BP_best_mse = np.inf
BP_best_weights = None
BP_best_convergence = None
BP_convergences = []
run_seeds = [r.randint(100, 900) for k in range(runs)]
print(run_seeds)

for run in range(runs):
    print(f"Starting run {run}")
    # Run GA and get model
    GA_weights, GA_convergence = GA(X, y, pop_size = pop_size, generations = generations, plot = False)
    GA_train_pred = forward_pass(X, GA_weights)
    # Evaluate GA performance and store as appropriate
    GA_mse = MSE(y, GA_train_pred)
    GA_mae = MAE(y, GA_train_pred)
    GA_vaf = VAF(y, GA_train_pred)
    GA_sse = SSE(y, GA_train_pred)
    GA_r2 = R2(y, GA_train_pred)
    GA_convergences.append(GA_convergence)

    if GA_mse < GA_best_mse:
        GA_best_weights = GA_weights
        GA_best_convergence = GA_convergence
        GA_best_run = run
        GA_best_mse = GA_mse
        GA_best_sse = GA_sse
        GA_best_r2 = GA_r2

    GA_mses.append(GA_mse)
    GA_maes.append(GA_mae)
    GA_vafs.append(GA_vaf)
    GA_sses.append(GA_sse)
    GA_r2s.append(GA_r2)

    # Run CSA and get model
    CSA_weights, CSA_convergence = CSA(X, y, pop_size = pop_size, generations = generations, plot = False)
    CSA_train_pred = forward_pass(X, CSA_weights)
    # Evaluate CSA performance and store as appropriate
    CSA_mse = MSE(y, CSA_train_pred)
    CSA_mae = MAE(y, CSA_train_pred)
    CSA_vaf = VAF(y, CSA_train_pred)
    CSA_sse = SSE(y, CSA_train_pred)
    CSA_r2 = R2(y, CSA_train_pred)
    CSA_convergences.append(CSA_convergence)

    if CSA_mse < CSA_best_mse:
        CSA_best_weights = CSA_weights
        CSA_best_convergence = CSA_convergence
        CSA_best_run = run
        CSA_best_mse = CSA_mse
        CSA_best_sse = CSA_sse
        CSA_best_r2 = CSA_r2

    CSA_mses.append(CSA_mse)
    CSA_maes.append(CSA_mae)
    CSA_vafs.append(CSA_vaf)
    CSA_sses.append(CSA_sse)
    CSA_r2s.append(CSA_r2)

    #Run BP
    BP_weights, BP_convergence = BP(X, y, epochs = generations)
    BP_train_pred = forward_pass(X, BP_weights)
    # Evaluate GA performance and store as appropriate
    BP_mse = MSE(y, BP_train_pred)
    BP_mae = MAE(y, BP_train_pred)
    BP_vaf = VAF(y, BP_train_pred)
    BP_sse = SSE(y, BP_train_pred)
    BP_r2 = R2(y, BP_train_pred)
    BP_convergences.append(BP_convergence)

    if BP_mse < BP_best_mse:
        BP_best_weights = BP_weights
        BP_best_convergence = BP_convergence
        BP_best_run = run
        BP_best_mse = BP_mse
        BP_best_sse = BP_sse
        BP_best_r2 = BP_r2

    BP_mses.append(BP_mse)
    BP_maes.append(BP_mae)
    BP_vafs.append(BP_vaf)
    BP_sses.append(BP_sse)
    BP_r2s.append(BP_r2)

# Run AR
AR_train_pred, coeffs = AR(y, lags=lags)
# Evaluate AR performance and store as appropriate
AR_mse = MSE(y[lags:], AR_train_pred)
AR_mae = MAE(y[lags:], AR_train_pred)
AR_vaf = VAF(y[lags:], AR_train_pred)
AR_sse = SSE(y[lags:], AR_train_pred)
AR_r2 = R2(y[lags:], AR_train_pred)


print(f"After {runs} runs")
print("==============================================================================")
print(" TRAINING PERFORMANCE")
print(f"GA PERFORMANCE:")
print(f"   Best Model== MSE: {GA_mses[GA_best_run]:.5f} | MAE: {GA_maes[GA_best_run]:.5f} | VAF: {GA_vafs[GA_best_run]:.5f} | SSE: {GA_sses[GA_best_run]:.5f} | R2: {GA_r2s[GA_best_run]:.5f}")
print(f"   Average===== MSE: {np.mean(GA_mses):.5f} | MAE: {np.mean(GA_maes):.5f} | VAF: {np.mean(GA_vafs):.5f} | SSE: {np.mean(GA_sses):.5f} | R2: {np.mean(GA_r2s):.5f}")
print(f"CSA PERFORMANCE:")
print(f"   Best Model== MSE: {CSA_mses[CSA_best_run]:.5f} | MAE: {CSA_maes[CSA_best_run]:.5f} | VAF: {CSA_vafs[CSA_best_run]:.5f} | SSE: {CSA_sses[CSA_best_run]:.5f} | R2: {CSA_r2s[CSA_best_run]:.5f}")
print(f"   Average===== MSE: {np.mean(CSA_mses):.5f} | MAE: {np.mean(CSA_maes)} | VAF: {np.mean(CSA_vafs):.5f} | SSE: {np.mean(CSA_sses):.5f} | R2: {np.mean(CSA_r2s):.5f}")
print(f"BP PERFORMANCE:")
print(f"   Best Model== MSE: {BP_mses[BP_best_run]:.5f} | MAE: {BP_maes[BP_best_run]:.5f} | VAF: {BP_vafs[BP_best_run]:.5f} | SSE: {BP_sses[BP_best_run]:.5f} | R2: {BP_r2s[BP_best_run]:.5f}")
print(f"   Average===== MSE: {np.mean(BP_mses):.5f} | MAE: {np.mean(BP_maes):.5f} | VAF: {np.mean(BP_vafs):.5f} | SSE: {np.mean(BP_sses):.5f} | R2: {np.mean(BP_r2s):.5f}")
print(f"AR PERFORMANCE:")
print(f"   AR Model== MSE: {AR_mse:.5f} | MAE: {AR_mae:.5f} | VAF: {AR_vaf:.5f} | SSE: {AR_sse:.5f} | R2: {AR_r2:.5f}")
print("==============================================================================")

# Compare Convergence
GA_average_convergence = np.mean(GA_convergences, axis = 0)
CSA_average_convergence = np.mean(CSA_convergences, axis = 0)
BP_average_convergence = np.mean(BP_convergences, axis = 0)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.plot(GA_best_convergence, label = 'Best')
plt.plot(GA_average_convergence, label = 'Average', linestyle = '-')
plt.title("GA Convergence")
plt.ylabel("MSE")
plt.xlabel("Generation")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(CSA_best_convergence, label = 'Best')
plt.plot(GA_average_convergence, label = 'Average', linestyle = '-')
plt.title("CSA Convergence")
plt.xlabel("Generation")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(BP_best_convergence, label = 'Best')
plt.plot(BP_average_convergence, label = 'Average', linestyle = '-')
plt.title("BP Convergence")
plt.xlabel("Generation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize = (8, 4))
plt.title("Compare Convergence")
plt.plot(GA_best_convergence, label = 'GA best')
plt.plot(CSA_best_convergence, label = 'CSA best')
plt.plot(BP_best_convergence, label = 'BP best')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Make predictions with the best performing model
print("Predictions Training Case")
print("Blue = Actual, Red = Predicted")
plt.figure(figsize=(10, 8))
pred = forward_pass(X, GA_best_weights) #GA
plt.subplot(2, 2, 1)
plt.plot(range(len(y)), y, label='Actual', color='black')
plt.plot(range(len(y)), pred, label = 'Predicted', linestyle = '--', color = 'green')
plt.ylabel("Target")
plt.title("GA")
plt.legend()
plt.grid(True)
pred = forward_pass(X, CSA_best_weights) #CSA
plt.subplot(2, 2, 2)
plt.plot(range(len(y)), y, label='Actual', color='black')
plt.plot(range(len(y)), pred, label = 'Predicted', linestyle = '--', color = 'orange')
plt.title("CSA")
plt.legend()
plt.grid(True)
pred = forward_pass(X, BP_best_weights) #BP
plt.subplot(2, 2, 3)
plt.plot(range(len(y)), y, label='Actual', color='black')
plt.plot(range(len(y)), pred, label = 'Predicted', linestyle = '--', color = 'blue')
plt.xlabel("Datapoints")
plt.ylabel("Target")
plt.title("BP")
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 4)
plt.plot(range(len(y[lags:])), y[lags:], label='Actual', color='black')
plt.plot(range(len(y[lags:])), AR_train_pred, label = 'Predicted', linestyle = '--', color = 'red')
plt.xlabel("Datapoints")
plt.title("AR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.title("Compare All")
plt.plot(range(len(y)), y, label='Actual', color='black')
pred = forward_pass(X, GA_best_weights)
plt.plot(range(len(y)), pred, label = 'GA Predicted', linestyle = '--', color = 'green')
pred = forward_pass(X, CSA_best_weights)
plt.plot(range(len(y)), pred, label = 'CSA Predicted', linestyle = '--', color = 'orange')
pred = forward_pass(X, BP_best_weights)
plt.plot(range(len(y)), pred, label = 'BP Predicted', linestyle = '--', color = 'blue')
plt.plot(range(lags, len(y)), AR_train_pred, label = 'AR_Predicted', linestyle = '--', color = 'red')
plt.legend()
plt.xlabel("Datapoints")
plt.ylabel("Target")
plt.grid(True)
plt.tight_layout()
plt.show()


# Make predictions with the best performing model
print("Predictions Training Case")
print("Blue = Actual, Red = Predicted")
plt.figure(figsize=(10, 4))
plt.subplot(2, 2, 1)
plt.plot(y, y, linewidth = .5, color = 'red')
pred = forward_pass(X, GA_best_weights)
plt.scatter(y, pred, label = 'GA', marker = 'o', facecolors = 'none', color = 'green', alpha = .7)
plt.ylabel("Predicted")
plt.title("GA")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(y, y, linewidth = .5, color = 'red')
pred = forward_pass(X, CSA_best_weights)
plt.scatter(y, pred, label = 'CSA', marker = 'o', color = 'orange', facecolors = 'none',  alpha = .7)
plt.title("CSA")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(y, y, linewidth = .5, color = 'red')
pred = forward_pass(X, BP_best_weights)
plt.scatter(y, pred, label = 'CSA', marker = 'o', color = 'blue', facecolors = 'none',  alpha = .7)
plt.xlabel("Target")
plt.ylabel("Predicted")
plt.title("BP")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(y, y, linewidth = .5, color = 'red')
plt.scatter(y[lags:], AR_train_pred, label = 'AR', marker = 'o', color = 'red', facecolors = 'none',  alpha = .7)
plt.xlabel("Target")
plt.title("AR")
plt.legend()
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

y = validation['Close']
y = y[1:].to_numpy()
X = validation[['Open', 'Close', 'High', 'Low']]
X = X[:-1].to_numpy()

s.transform(X)

GA_val = forward_pass(X, GA_best_weights)
CSA_val = forward_pass(X, CSA_best_weights)
BP_val = forward_pass(X, BP_best_weights)
AR_val = validate_AR(y, coeffs)

print("==============================================================================")
print(F"VALIDATION PERFORMANCE")
print(f"GA PERFORMANCE")
print(f"   MSE {MSE(y, GA_val):.5f} | MAE: {MAE(y, GA_val):.5f} | VAF: {VAF(y, GA_val):.5f} | SSE: {SSE(y, GA_val):.5f} | R2: {R2(y, GA_val):.5f}")
print(f"CSA PERFORMANCE:")
print(f"   MSE {MSE(y, CSA_val):.5f} | MAE: {MAE(y, CSA_val):.5f} | VAF: {VAF(y, CSA_val):.5f} | SSE: {SSE(y, CSA_val):.5f} | R2: {R2(y, CSA_val):.5f}")
print(f"BP PERFORMANCE:")
print(f"   MSE {MSE(y, BP_val):.5f} | MAE: {MAE(y, BP_val):.5f} | VAF: {VAF(y, BP_val):.5f} | SSE: {SSE(y, BP_val):.5f} | R2: {R2(y, BP_val):.5f}")
print(f"AR PERFORMANCE")
print(f"   MSE {MSE(y[lags:], AR_val):.5f} | MAE: {MAE(y[lags:], AR_val):.5f} | VAF: {VAF(y[lags:], AR_val):.5f} | SSE: {SSE(y[lags:], AR_val):.5f} | R2: {R2(y[lags:], AR_val):.5f}")
print("==============================================================================")

# Make predictions with the best performing model
print("Predictions Testing Case")
plt.figure(figsize=(10, 8))

pred = forward_pass(X, GA_best_weights)
plt.subplot(2, 2, 1)
plt.plot(range(len(y)), y, label='Actual', color='black')
plt.plot(range(len(y)), pred, label = 'Predicted', linestyle = '--', color = 'green')
plt.ylabel("Target")
plt.title("GA")
plt.legend()
plt.grid(True)

pred = forward_pass(X, CSA_best_weights)
plt.subplot(2, 2, 2)
plt.plot(range(len(y)), y, label='Actual', color='black')
plt.plot(range(len(y)), pred, label = 'Predicted', linestyle = '--', color = 'orange')
plt.title("CSA")
plt.legend()
plt.grid(True)

pred = forward_pass(X, BP_best_weights)
plt.subplot(2, 2, 3)
plt.plot(range(len(y)), y, label='Actual', color='black')
plt.plot(range(len(y)), pred, label = 'Predicted', linestyle = '--', color = 'blue')
plt.ylabel("Target")
plt.xlabel("Datapoints")
plt.title("BP")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(range(len(y[lags:])), y[lags:], label='Actual', color='black')
plt.plot(range(len(y[lags:])), AR_val, label = 'Predicted', linestyle = '--', color = 'red')
plt.xlabel("Datapoints")
plt.title("AR")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.title("Compare ALL")
plt.plot(range(len(y)), y, label='Actual', color='black')
pred = forward_pass(X, GA_best_weights)
plt.plot(range(len(y)), pred, label = 'GA Predicted', linestyle = '--', color = 'green')
pred = forward_pass(X, CSA_best_weights)
plt.plot(range(len(y)), pred, label = 'CSA Predicted', linestyle = '--', color = 'orange')
pred = forward_pass(X, BP_best_weights)
plt.plot(range(len(y)), pred, label = 'BP Predicted', linestyle = '--', color = 'blue')
plt.plot(range(lags, len(y)), AR_val, label = 'AR_Predicted', linestyle = '--', color = 'red')
plt.legend()
plt.xlabel("Datapoints")
plt.ylabel("Target")
plt.grid(True)
plt.tight_layout()
plt.show()


# Validation
print("Predictions Validation Case")
print("Blue = Actual, Red = Predicted")
plt.figure(figsize=(10, 4))
plt.subplot(2, 2, 1)
plt.plot(y, y, linewidth = .5, color = 'red')
pred = forward_pass(X, GA_best_weights)
plt.scatter(y, pred, label = 'GA', marker = 'o', facecolors = 'none', color = 'green', alpha = .7)
plt.ylabel("Predicted")
plt.title("GA")

plt.subplot(2, 2, 2)
plt.plot(y, y, linewidth = .5, color = 'red')
pred = forward_pass(X, CSA_best_weights)
plt.scatter(y, pred, label = 'CSA', marker = 'o', color = 'orange', facecolors = 'none',  alpha = .7)
plt.title("CSA")

plt.subplot(2, 2, 3)
plt.plot(y, y, linewidth = .5, color = 'red')
pred = forward_pass(X, BP_best_weights)
plt.scatter(y, pred, label = 'BP', marker = 'o', color = 'blue', facecolors = 'none',  alpha = .7)
plt.xlabel("Target")
plt.ylabel("Predicted")
plt.title("BP")

plt.subplot(2, 2, 4)
plt.plot(y, y, linewidth = .5, color = 'red')
plt.scatter(y[lags:], AR_val, label = 'AR', marker = 'o', color = 'red', facecolors = 'none',  alpha = .7)
plt.xlabel("Target")
plt.title("AR")
plt.legend()
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()
