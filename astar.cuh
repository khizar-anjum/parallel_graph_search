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

#endif