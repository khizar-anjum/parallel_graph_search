#include "dijkstra.cuh"
//#include "pqueue.cu"

int* dijkstra_par(graph &g, int src){
	// starts at the source and calculates the distance for all the 
	// vertices inside the graph

	// check if src exists and populate it
	std::map<int, int>::iterator it = g.name_to_vertex.find(src);
	if(it == g.name_to_vertex.end()) throw std::invalid_argument( "Invalid vertex name" );
	int src_vertex = it->second;

	// initialize everything
	bool finished = false;
	int NUM_QUEUES = 1;
	bool* visited = new bool[g.num_vertices];
	int* costs = new int[g.num_vertices];
	int* size_h = new int[NUM_QUEUES];
	unsigned int* SWsize; 
	unsigned int* Tsize;
	unsigned int* SWsize_h = new unsigned int[1]; SWsize_h[0] = 0; 
	unsigned int* Tsize_h = new unsigned int[1]; Tsize_h[0] = 0;
	for(int i = 0; i < g.num_vertices; i++){
		visited[i] = false; //visited flags for vertices
		costs[i] = 0x70000000; //costs for each vertex
	}
	costs[src_vertex] = 0;
	for(int i = 0; i < NUM_QUEUES; i++){
		size_h[i] = 0;
	}


	int* weight_arr_d; int* connected_to_d; int* costs_d; 
	size_t bytes = g.num_edges * sizeof(int);
	cudaMalloc(&weight_arr_d, bytes);
	cudaMalloc(&connected_to_d, bytes);

	int* heap; int* size; int* S; int* W; int* T; 
	bool* visited_d; int* num_connected_d; int* index_arr_d;
	size_t vbytes = g.num_vertices * sizeof(int);
	cudaMalloc(&heap, 2*vbytes*NUM_QUEUES);
	cudaMalloc(&size, NUM_QUEUES*sizeof(int));
	cudaMalloc(&num_connected_d, vbytes);
	cudaMalloc(&index_arr_d, vbytes);
	cudaMalloc(&costs_d, vbytes);
	cudaMalloc(&S, vbytes);
	cudaMalloc(&W, vbytes);
	cudaMalloc(&T, vbytes);
	cudaMalloc(&SWsize, sizeof(unsigned int));
	cudaMalloc(&Tsize, sizeof(unsigned int));
	cudaMalloc(&visited_d, g.num_vertices*sizeof(bool));

	//lets copy all the stuff from device memory to host memory
	cudaMemcpy(size, size_h, NUM_QUEUES*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(weight_arr_d, g.weight_arr, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(index_arr_d, g.index_arr, vbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(connected_to_d, g.connected_to, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(costs_d, costs, vbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(visited_d, visited, g.num_vertices*sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(num_connected_d, g.num_connected, vbytes, cudaMemcpyHostToDevice);
	insertKernelDijkstra <<<1,1>>> (heap, size, src_vertex, 0, g.num_vertices);
	
	//Now, extracting mins and carrying on till we find the destination!
	while(!finished){
		//Extracting min and pulling the corresponding vertex into cloud
		cudaMemcpy(SWsize, SWsize_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
		extractKernelDijkstra<<<1, NUM_QUEUES>>>(heap, size, S, W, SWsize, num_connected_d, weight_arr_d,
			costs_d, index_arr_d, connected_to_d, NUM_QUEUES, g.num_vertices);
		cudaDeviceSynchronize();

		cudaMemcpy(Tsize, Tsize_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
		deduplicateKernelDijkstra<<<1, NUM_QUEUES>>>(S, W, SWsize, T, Tsize, visited_d, costs_d, g.num_vertices);
		cudaDeviceSynchronize();

		computeKernelDijkstra<<<1, NUM_QUEUES>>>(heap, size, T, Tsize, visited_d, costs_d, g.num_vertices);
		cudaDeviceSynchronize();

		//Exit when all the vertices explored 
		//meaning that all queues are empty
		cudaMemcpy(size_h, size, NUM_QUEUES*sizeof(int), cudaMemcpyDeviceToHost);
		for(int i = 0; i < NUM_QUEUES; i++){
			//printf("%d\n", size_h[i]);
			if(size_h[i] > 1)
				break;	
			finished = true;
		}
	}

	cudaMemcpy(costs, costs_d, vbytes, cudaMemcpyDeviceToHost);
	cudaFree(heap); cudaFree(size); cudaFree(num_connected_d); cudaFree(index_arr_d);
	cudaFree(costs_d); cudaFree(S); cudaFree(W); cudaFree(T); cudaFree(visited_d);
	cudaFree(weight_arr_d); cudaFree(connected_to_d); cudaFree(SWsize); cudaFree(Tsize);
	delete [] visited;

	return costs;
}

__global__ void insertKernelDijkstra(int* heap, int* size, int src_vertex, int priority, int num_vertices){
	//kernel to insert elements
	insert_GPU(heap, size[0], src_vertex, priority, num_vertices);
}

__global__ void extractKernelDijkstra(int* heap, int* size, int* S, int* W, unsigned int* SWsize, int* num_connected_d, int* weight_arr_d,
								int* costs_d, int* index_arr_d, int* connected_to_d, int NUM_QUEUES, int num_vertices){
	//kernel to extract elements from queues. Here num_threads must be equal to NUM_QUEUES
	int current_vertex;
	int current_weight;
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int oldSWsize = 0;
	if(size[index] > 0){
		ExtractMin_GPU(&heap[2*index*num_vertices], size[index], current_vertex, current_weight, num_vertices);
		//printf("%d %d\n", current_vertex, current_weight);
		for(int i = 0; i < num_connected_d[current_vertex]; i++){
			oldSWsize = atomicAdd(SWsize, 1);
			atomicExch(&S[oldSWsize], connected_to_d[index_arr_d[current_vertex] + i]);
			W[oldSWsize] = weight_arr_d[index_arr_d[current_vertex] + i] + costs_d[current_vertex];
		}
	}
}

__global__ void deduplicateKernelDijkstra(int* S, int* W, unsigned int* SWsize, int* T, unsigned int* Tsize, bool* visited_d, int* costs_d, int num_vertices){
	//kernel to deduplicate elements in S. 
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	unsigned int oldTsize;
	
	for(int i = index; i < SWsize[0]; i+=stride){
		//printf("%d %d %d %d %d\n", i, S[i], visited_d[S[i]], costs_d[S[i]], W[i]);
		if((visited_d[S[i]] == true) && (costs_d[S[i]] <= W[i])){
			continue;
		}
		else{
			oldTsize = atomicAdd(Tsize, 1);
			T[oldTsize] = S[i];
			if(costs_d[S[i]] > W[i]){
				atomicExch(&costs_d[S[i]], W[i]);
			}
		}
	}
}

__global__ void computeKernelDijkstra(int* heap, int* size, int* T, unsigned int* Tsize, bool* visited_d, int* costs_d, int num_vertices){
	//kernel to computee elements in T. Since this accesses queues, num_threads must be equal to NUM_QUEUES
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	//printf("%d\n", Tsize[0]);
	for(int i = index; i < Tsize[0]; i+= stride){
		insert_GPU(&heap[2*index*num_vertices], size[index], T[i], costs_d[T[i]], num_vertices);
		visited_d[T[i]] = true;
	}
}
