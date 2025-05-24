import sys
import random
import time
import cupy as cp

# Broj cvorova u grafu
V = 10000

# SEKVENCIJALI ALGORITAM
def min_keySEQ(key, visited): #argumenti su niz kljuceva i niz visited
    min_val = sys.maxsize 
    min_index = -1
   
    for v in range(V):
        if not visited[v] and key[v] < min_val:
            min_val = key[v]
            min_index = v
     
    return min_index #vraca indeks minimalnog elementa iz niza kljuceva

def prim_mstSEQ(graph):
    from_vertex = [-1] * V  # niz za cuvanje konstruisanog MST
    key = [sys.maxsize] * V  # vrijednosti kljuceva za odabir minimalne grane
    visited = [False] * V  # neposjeceni cvorovi u MST

    key[0] = 0  # pocinjemo od prvog cvora

    for _ in range(V - 1): 
        u = min_keySEQ(key, visited)  # biramo cvor sa najmanjom vrijednoscu kljuca (tj biramo najlaksu granu)
        visited[u] = True
        
        for v in range(V):
            if graph[u][v] and not visited[v] and graph[u][v] < key[v]:
                from_vertex[v] = u
                key[v] = graph[u][v]
    
    print_mst(from_vertex, graph)


def print_mst(from_vertex, graph):
    total_cost = 0.0
    edge_count = 0
    #print("Edge   Weight")
    for i in range(1, V):
        total_cost = total_cost + graph[i][from_vertex[i]]
        edge_count += 1
        #print(f"{from_vertex[i]} - {i}    {graph[i][from_vertex[i]]}")
    print("Edge count: ", edge_count)
    print("Total cost: " , total_cost)

###PARALELIZOVANI

def min_key(key, visited):
    min_index = cp.argmin(cp.where(visited == 0, key, cp.inf)) # biramo cvor sa najmanjom vrijednoscu kljuca (tj biramo najlaksu granu)
    return int(min_index)

def prim_mst(graph):
    from_vertex = cp.full(V, -1, dtype=cp.int32)
    key = cp.full(V, cp.inf, dtype=cp.float64)
    visited = cp.zeros(V, dtype=cp.int32)

    key[0] = 0  # pocinjemo od prvog cvora 

    for _ in range(V - 1):
        u = min_key(key, visited)
        visited[u] = 1

        # GPU paralelne operacije za update niza kljuceva
        mask = (graph[u] > 0) & (visited == 0) & (graph[u] < key)
        from_vertex = cp.where(mask, u, from_vertex)
        key = cp.where(mask, graph[u], key)

    print_mst(from_vertex,graph)


def main():

    graph = cp.random.randint(0, 10, (V, V), dtype=cp.int32)
    cp.fill_diagonal(graph, 0)
    graph = (graph + graph.T) / 2
    graph = graph + graph
    #print(graph)
    graph_seq = graph.tolist()

    start_time = time.time()
    prim_mstSEQ(graph_seq)
    end_time = time.time()
    print(f"Time for seq = {end_time - start_time:.6f} seconds")
    
    start_time = time.time()
    prim_mst(graph)
    end_time = time.time()
    print(f"CUDA Accelerated Time = {end_time - start_time:.6f} seconds")
    
    # graph_seq1 = [[random.randint(1, 9) for _ in range(V)] for _ in range(V)]
    # for i in range(V):
    #     graph_seq1[i][i] = 0
    # for i in range(V):
    #     for j in range(V):
    #         graph_seq1[j][i] = graph_seq1[i][j]
    # graph1 = cp.array(graph_seq1)
    
    # start_time = time.time()
    # prim_mstSEQ(graph_seq1)
    # end_time = time.time()
    # print(f"Time for seq = {end_time - start_time:.6f} seconds")
    
    # start_time = time.time()
    # prim_mst(graph1)
    # end_time = time.time()
    # print(f"CUDA Accelerated Time = {end_time - start_time:.6f} seconds")

   
if __name__ == "__main__":
    main()

