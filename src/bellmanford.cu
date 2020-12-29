#include "bellmanford.cuh"

int* bellmanford_par(graph &g, int src, int BLOCK_SIZE, int GRID_SIZE){
	// starts at the source and calculates the distance for all the 
	// vertices inside the graph
	// check if src exists and populate it
	std::map<int, int>::iterator it = g.name_to_vertex.find(src);
	if(it == g.name_to_vertex.end()) throw std::invalid_argument( "Invalid vertex name" );
	int src_vertex = it->second;
	bool nochange = true;

	int* costs = new int[g.num_vertices];
	int* connected_from = new int[g.num_edges];
	// lets initialize these costs to infinity (A high enough number)
	// also initialize the connected_from array
	for(int i = 0; i < g.num_vertices; i++){
		costs[i] = 0x70000000;
		for(int j = 0; j < g.num_connected[i]; j++){
			connected_from[g.index_arr[i] + j] = i;
		}
	}
	costs[src_vertex] = 0;

	int* weight_arr_d; int* connected_to_d; int* connected_from_d; int* costs_d;
	bool* nochange_d;
	size_t bytes = g.num_edges * sizeof(int);
	size_t vbytes = g.num_vertices * sizeof(int);
	cudaMalloc(&weight_arr_d, bytes);
	cudaMalloc(&connected_to_d, bytes);
	cudaMalloc(&connected_from_d, bytes);
	cudaMalloc(&costs_d, vbytes); //
	cudaMalloc(&nochange_d, sizeof(bool));

	//lets copy all the stuff from device memory to host memory
	cudaMemcpy(weight_arr_d, g.weight_arr, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(connected_to_d, g.connected_to, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(connected_from_d, connected_from, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(costs_d, costs, vbytes, cudaMemcpyHostToDevice);

	bool* placeholder = new bool[1];

	// lets start crunching
	for(int j = 0; j < g.num_vertices ; j++){
		//Repeating for |V| - 1 times 
		cudaMemcpy(nochange_d, &nochange, sizeof(bool), cudaMemcpyHostToDevice);
		bellmanFord_kernel <<< GRID_SIZE, BLOCK_SIZE >>> (weight_arr_d, connected_to_d, connected_from_d, costs_d, g.num_edges, nochange_d);
		cudaDeviceSynchronize();
		cudaMemcpy(placeholder, nochange_d, sizeof(bool), cudaMemcpyDeviceToHost);
		if(placeholder[0]){
			break;
		}
	}
	checkNegativeCycle_kernel <<< GRID_SIZE, BLOCK_SIZE >>> (weight_arr_d, connected_to_d, connected_from_d, costs_d, g.num_edges);
	cudaDeviceSynchronize();
	
	cudaMemcpy(costs, costs_d, vbytes, cudaMemcpyDeviceToHost);
	if(costs[0] == -1){ // Error checking
		throw std::logic_error( "Graph contains a negative-weight cycle" );
	}
	cudaFree(weight_arr_d); cudaFree(connected_to_d); cudaFree(connected_from_d); cudaFree(costs_d);
	cudaFree(nochange_d);
	delete [] placeholder;
	delete [] connected_from;
	return costs;
}

__global__ void bellmanFord_kernel(int* weight_arr_d, int* connected_to_d, int* connected_from_d, int* costs_d, int num_edges, bool* nochange){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	//Run through every edge and update the distances!
	for(int i = index; i < num_edges; i+=stride){
		//lets see if we can update this
		//printf("%d\n", costs_d[connected_to_d[i]]);
		if(costs_d[connected_to_d[i]] > weight_arr_d[i] + costs_d[connected_from_d[i]]){
			nochange[0] = false;
			atomicExch(&costs_d[connected_to_d[i]], weight_arr_d[i] + costs_d[connected_from_d[i]]);
		}
	}
}

__global__ void checkNegativeCycle_kernel(int* weight_arr_d, int* connected_to_d, int* connected_from_d, int* costs_d, int num_edges){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	//Run through every edge and update the distances!
	for(int i = index; i < num_edges; i+=stride){
		//lets see if we can update this
		if(costs_d[connected_to_d[i]] > weight_arr_d[i] + costs_d[connected_from_d[i]]){
			atomicExch(&costs_d[0], -1);
			return;
		}
	}
}