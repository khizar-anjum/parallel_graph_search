#ifndef ASTAR__CUH
#define ASTAR__CUH

#include "graph.h"
#include "pqueue.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

int* astar_par(graph &g, int src, int (*h)(int));
std::vector<int> astar_par(graph &g, int src, int dst, int &cost, int (*h)(int));
std::vector<int> dijkstra_par(graph &g, int src, int dst, int &cost);
int* dijkstra_par(graph &g, int src);

__global__ void insertKernelDijkstra(int* heap, int* size, int src_vertex, int priority, int num_vertices);
__global__ void extractKernelDijkstra(int* heap, int* size, int* S, int* W, unsigned int* SWsize, int* num_connected_d, int* weight_arr_d,
								int* costs_d, int* index_arr_d, int* connected_to_d, int NUM_QUEUES, int num_vertices);
__global__ void deduplicateKernelDijkstra(int* S, int* W, unsigned int* SWsize, int* T, unsigned int* Tsize, bool* visited_d, int* costs_d, int num_vertices);
__global__ void computeKernelDijkstra(int* heap, int* size, int* T, unsigned int* Tsize, bool* visited_d, int* costs_d, int num_vertices);

#endif