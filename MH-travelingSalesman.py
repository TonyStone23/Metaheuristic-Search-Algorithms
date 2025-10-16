#============================================================================================
# MH search algorithms for the Traveling Salesman Problem
#============================================================================================
import networkx as nx
import numpy as np
import pandas as dp
import random as r
import matplotlib.pyplot as plt
r.seed(42)
np.random.seed(42)

def TSP(numCities):
    matrix = [[r.uniform(5, 20) if i != j else 0 for j in range(numCities)]for i in range(numCities)]
    return np.asarray(matrix, dtype = np.float32)

def showTSP(tsp, labels = False, arrows = False):
    G = nx.Graph()

    # --- Add nodes
    G.add_nodes_from(range(len(tsp)))

    # --- Add edges with weights from the matrix
    for i in range(len(tsp)):
        for j in range(len(tsp)):
            if i != j:
                G.add_edge(i, j, weight=tsp[i][j])

    # --- Layout for node positions
    pos = nx.spring_layout(G, seed=42)

    # --- Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10)
    
    if labels == True:
        edge_labels = {(i, j): f"{tsp[i][j]:.1f}" for i in range(len(tsp)) for j in range(len(tsp)) if i != j}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Connected Graph of TSP Cities")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return

def plotbest(route):

    return

def populationCost(population, tsp):
    next_city = np.roll(population, -1, axis = 1)
    weights = tsp[population, next_city]
    return weights.sum(axis=1)

def GA(tsp, pop_size, num_cities, generations, mutation_rate = .01):
    print("Starting GA")
    ind = [num for num in range(0, num_cities)]
    population = np.asarray([np.random.permutation(ind) for i in range(pop_size)])
    best_scores = []
    best_score = np.inf
    best_route = None

    for gen in range (generations):
        scores = populationCost(population, tsp)
        best_idx = scores.argmin()
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_route = population[best_idx].tolist()
        best_scores.append(best_score)

        top_idx = scores.argsort()[:(pop_size // 5)]
        parents = population[top_idx]

        new_population = []
        for _ in range(pop_size):
            #Reproduction and mutation
            p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
            point = r.randint(0, num_cities-1)
            child = p1[:point]
            childfill = p2[~np.isin(p2, child)]
            child = np.concatenate([child, childfill])
            if r.random() < mutation_rate:
                r1, r2 = r.randint(0, num_cities-1), r.randint(0, num_cities-1)
                child[r1], child[r2] = child[r2], child[r1]
            new_population.append(child)

        population = np.vstack(new_population)
    return best_scores, best_score, best_route

def ABC():
    return

def ACO(tsp, pop_size, num_cities, generations, q = 100, a = 1, b = 2, rh = 0.1):
    print("Starting ACO")
    pheromone = np.ones((num_cities, num_cities))
    desire = 1/(tsp + 1e-10) 
    best_scores = []
    best_score = np.inf
    best_route = None

    for gen in range(generations):
        routes = []

        for ant in range(pop_size):
            visited = [r.randint(0, num_cities-1)]
            for _ in range(num_cities-1):
                current = visited[-1]
                unvisited = list(set(range(num_cities)) - set(visited))
                probs = []

                for j in unvisited:
                    t = pheromone[current][j] ** a
                    e = desire[current][j] ** b
                    probs.append(t * e)

                probs = np.array(probs)
                probs /= probs.sum()
                next_city = np.random.choice(unvisited, p=probs)
                visited.append(next_city)

            routes.append(visited)
        routes_arr = np.asarray(routes)

        costs_arr = populationCost(routes_arr, tsp)
        best_idx = costs_arr.argmin()

        if costs_arr[best_idx] < best_score:
            best_score = costs_arr[best_idx]
            best_route = routes_arr[best_idx].tolist()
        best_scores.append(best_score)

        pheromone *= (1 - rh)

        # Pheromone update
        for route, cost in zip(routes, costs_arr):
            for i in range(num_cities):
                a = route[i]
                b = route[(i + 1) % num_cities]
                pheromone[a][b] += q / cost

    return best_scores, best_score, best_route

def GTO(tsp, pop_size, num_cities, generations, p=0.03, beta=3, w=0.8):
    print("Starting GTO")
    ind = [num for num in range(0, num_cities)]
    population = np.asarray([np.random.permutation(ind) for i in range(pop_size)])
    best_scores = []
    best_score = np.inf
    best_route = None
    
    for gen in range(generations):
        scores = populationCost(population, tsp)
        
        best_idx = scores.argmin()
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_route = population[best_idx].tolist()
        best_scores.append(best_score)
        
        silverback = population[best_idx]
        sorted_idx = scores.argsort()
        sorted_population = population[sorted_idx]
        
        new_population = []
        
        for i in range(pop_size):
            current_gorilla = population[i]
            
            if r.random() < p:
                new_gorilla = np.random.permutation(ind)
            else:
                if r.random() < 0.5:
                    new_gorilla = current_gorilla.copy()
                    for j in range(num_cities):
                        if r.random() < w:
                            silverback_city = silverback[j]
                            if new_gorilla[j] != silverback_city:
                                current_pos = np.where(new_gorilla == silverback_city)[0][0]
                                new_gorilla[j], new_gorilla[current_pos] = new_gorilla[current_pos], new_gorilla[j]
                    
                else:
                    if i < pop_size // 2:
                        competitor_idx = r.randint(0, pop_size // 2 - 1)
                        competitor = sorted_population[competitor_idx]
                        new_gorilla = current_gorilla.copy()
                        crossover_points = sorted(r.sample(range(num_cities), 2))
                        start, end = crossover_points[0], crossover_points[1]
                        segment = competitor[start:end]
        
                        mask = np.isin(new_gorilla, segment, invert=True)
                        remaining_cities = new_gorilla[mask]

                        new_route = np.zeros(num_cities, dtype=int)
                        remaining_idx = 0
                        
                        for j in range(num_cities):
                            if start <= j < end:
                                new_route[j] = segment[j - start]
                            else:
                                new_route[j] = remaining_cities[remaining_idx]
                                remaining_idx += 1
                        
                        new_gorilla = new_route
                        
                    else:
                        adult_idx = r.randint(0, pop_size // 2 - 1)
                        adult = sorted_population[adult_idx]
                        
                        new_gorilla = current_gorilla.copy()
                        if r.random() < 0.7:
                            seq_length = r.randint(2, min(5, num_cities // 2))
                            start_pos = r.randint(0, num_cities - seq_length)
                            
                            adult_subsequence = adult[start_pos:start_pos + seq_length]
                            mask = np.isin(new_gorilla, adult_subsequence, invert=True)
                            filtered_gorilla = new_gorilla[mask]
                            insert_pos = r.randint(0, len(filtered_gorilla))
                            new_route = np.concatenate([filtered_gorilla[:insert_pos], adult_subsequence, filtered_gorilla[insert_pos:]])
                            
                            new_gorilla = new_route
            
            if r.random() < 0.1:
                if num_cities > 3:
                    i_local, j_local = sorted(r.sample(range(num_cities), 2))
                    new_gorilla[i_local:j_local+1] = new_gorilla[i_local:j_local+1][::-1]
            
            new_population.append(new_gorilla)
        
        population = np.vstack(new_population)
    
    return best_scores, best_score, best_route

Cs = 20
generations = 100
pop_size = 200

tsp = TSP(Cs)
showTSP(tsp)

GA_best_scores, GA_best_score, GA_best_route = GA(tsp, pop_size, Cs, generations)
ACO_best_scores, ACO_best_score, ACO_best_route = ACO(tsp, pop_size, Cs, generations)
GTO_best_scores, GTO_best_score, GTO_best_route = GTO(tsp, pop_size, Cs, generations)

print(f"GA best score: {GA_best_score} | Route: {GA_best_route}")
print(f"ACO best score: {ACO_best_score} | Route: {ACO_best_route}")
print(f"ACO best score: {ACO_best_score} | Route: {ACO_best_route}")

plt.figure(figsize = (8, 6))
plt.plot(range(0, generations), GA_best_scores, color = 'green', label = 'GA')
plt.plot(range(0, generations), ACO_best_scores, color = 'maroon', label = 'ACO')
plt.plot(range(0, generations), GTO_best_scores, color = 'gray', label = 'GTO')

plt.title("Convergence")
plt.legend()
plt.tight_layout
plt.xlabel("Generations")
plt.ylabel("Travel Cost")
plt.show()

