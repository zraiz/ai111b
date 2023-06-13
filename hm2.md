# 110911542 Homework2

The Traveling Salesperson Problem (TSP) is a well-known NP-hard problem. Approximate solutions to TSP can be provided by algorithms such as the Nearest Neighbor algorithm and the 2-Opt algorithm. The Nearest Neighbor algorithm is simple and efficient, but may not always provide optimal solutions. The 2-Opt algorithm iteratively improves the initial solution by swapping pairs of edges to reduce the total distance. There are also more advanced algorithms available, such as the Lin-Kernighan heuristic, Ant Colony Optimization, and Genetic Algorithms, which can provide better solutions for larger problem sizes. The quality of the solution obtained depends on the complexity of the TSP instance.
> this is the code
```
import math

def nearest_neighbor(cities):
    num_cities = len(cities)
    visited = [False] * num_cities
    path = []
    
    # Start with the first city as the current city
    current_city = 0
    path.append(current_city)
    visited[current_city] = True
    
    # Visit the remaining cities
    for _ in range(num_cities - 1):
        nearest_city = None
        nearest_dist = math.inf
        
        # Find the nearest unvisited city
        for i in range(num_cities):
            if not visited[i]:
                distance = math.dist(cities[current_city], cities[i])
                if distance < nearest_dist:
                    nearest_dist = distance
                    nearest_city = i
        
        # Move to the nearest city
        current_city = nearest_city
        path.append(current_city)
        visited[current_city] = True
    
    # Return to the starting city
    path.append(0)
    
    return path

# Example usage
cities = [(0, 0), (1, 1), (2, 2), (3, 3)]
solution = nearest_neighbor(cities)
print(solution)
```
> ChatGPT (directly ask the code to solve TSP using simulated annealing method, no change)
