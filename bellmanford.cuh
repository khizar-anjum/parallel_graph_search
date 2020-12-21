#ifndef BELLMANFORD__CUH
#define BELLMANFORD__CUH

#include "graph.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

__global__ void bellmanFord_kernel(int* weight_arr_d, int* connected_to_d, int* connected_from_d, int* costs, int num_edges);
__global__ void checkNegativeCycle_kernel(int* weight_arr_d, int* connected_to_d, int* connected_from_d, int* costs, int num_edges);


int* bellmanford_par(graph &g, int src);

#endif