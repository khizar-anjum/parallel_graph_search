#include "astar.h"

int* astar_seq(graph &g, int src, int (*h)(int)){
	// starts at the source and calculates the distance for all the 
	// vertices inside the graph
	// here h is the heuristic function
	// initialize heap and its size
	int* heap = new int[2*g.num_vertices];
	int* costs = new int[g.num_vertices];
	int size = 0;

	// check if src exists and populate it
	std::map<int, int>::iterator it = g.name_to_vertex.find(src);
	if(it == g.name_to_vertex.end()) throw std::invalid_argument( "Invalid vertex name" );
	int src_vertex = it->second;

	int current_vertex; //current vertex being explored
	int current_weight; //current weight
	int next_vertex; //next vertex to explore
	int next_weight; //next weight
	int next_pq_index; //next vertex index in priority queue
	bool visited[g.num_vertices] = {false}; //visited flags for vertices

	insert(heap, size, src_vertex, (*h)(src_vertex), g.num_vertices);
	
	//Now, extracting mins and carrying on till we find the destination!
	while(1){
		//Extracting min and pulling the corresponding vertex into cloud
		ExtractMin(heap, size, current_vertex, current_weight, g.num_vertices);
		current_weight -= (*h)(current_vertex);
		costs[current_vertex] = current_weight;
		visited[current_vertex] = true;

		//Run through every edge and update the distances!
		for(int i = 0; i < g.num_connected[current_vertex]; i++){
			next_vertex = g.connected_to[g.index_arr[current_vertex] + i];
			if(!visited[next_vertex]){
				// if not visited, lets see if we can update the distance
				getItem(heap, size, next_vertex, next_pq_index, next_weight, g.num_vertices);
				if(next_pq_index == -1){ //if not found in pqueue
					next_weight = current_weight + g.weight_arr[g.index_arr[current_vertex] + i] + (*h)(next_vertex);
					insert(heap, size, next_vertex, next_weight, g.num_vertices);
				}
				else if(next_weight > current_weight + g.weight_arr[g.index_arr[current_vertex] + i]){
					// if found but a better weight available
					next_weight = current_weight + g.weight_arr[g.index_arr[current_vertex] + i] + (*h)(next_vertex);
					// remove the old entry and add new one 
					remove(heap, size, next_pq_index, g.num_vertices);
					insert(heap, size, next_vertex, next_weight, g.num_vertices);
				}
			}
		}

		//Store current into source (which is prev)!!
		if(size < 1){
			return costs;
		}
	}

	return costs;
}


std::vector<int> astar_seq(graph &g, int src, int dst, int &cost, int (*h)(int)){
	// here h is the heuristic function
	// initialize heap and its size
	int* heap = new int[2*g.num_vertices];
	int size = 0;

	// check if src exists and populate it
	std::map<int, int>::iterator it = g.name_to_vertex.find(src);
	if(it == g.name_to_vertex.end()) throw std::invalid_argument( "Invalid vertex name" );
	int src_vertex = it->second;
	// check if dst exists and populate it
	it = g.name_to_vertex.find(dst);
	if(it == g.name_to_vertex.end()) throw std::invalid_argument( "Invalid vertex name" );
	int dst_vertex = it->second;

	int current_vertex; //current vertex being explored
	int current_weight; //current weight
	int next_vertex; //next vertex to explore
	int next_weight; //next weight
	int next_pq_index; //next vertex index in priority queue
	std::vector<int> fullpath; //full path of vertices
	bool visited[g.num_vertices] = {false}; //visited flags for vertices
	int prev[g.num_vertices] = {0}; //used for backtracking at the end

	insert(heap, size, src_vertex, (*h)(src_vertex), g.num_vertices);
	prev[src_vertex] = src_vertex;
	
	//Now, extracting mins and carrying on till we find the destination!
	while(1){
		//Extracting min and pulling the corresponding vertex into cloud
		ExtractMin(heap, size, current_vertex, current_weight, g.num_vertices);
		current_weight -= (*h)(current_vertex);
		visited[current_vertex] = true;
		//	current->prev = source;
		//	cout << "current is " << current->name << " its dist is " << current->dist << endl;
		//Destination found mechanism!
		if(current_vertex == dst_vertex){
			break;
		}

		//Run through every edge and update the distances!
		for(int i = 0; i < g.num_connected[current_vertex]; i++){
			next_vertex = g.connected_to[g.index_arr[current_vertex] + i];
			if(!visited[next_vertex]){
				// if not visited, lets see if we can update the distance
				getItem(heap, size, next_vertex, next_pq_index, next_weight, g.num_vertices);
				if(next_pq_index == -1){ //if not found in pqueue
					next_weight = current_weight + g.weight_arr[g.index_arr[current_vertex] + i] + (*h)(next_vertex);
					insert(heap, size, next_vertex, next_weight, g.num_vertices);
					prev[next_vertex] = current_vertex;
				}
				else if(next_weight > current_weight + g.weight_arr[g.index_arr[current_vertex] + i]){
					// if found but a better weight available
					next_weight = current_weight + g.weight_arr[g.index_arr[current_vertex] + i] + (*h)(next_vertex);
					// remove the old entry and add new one 
					remove(heap, size, next_pq_index, g.num_vertices);
					insert(heap, size, next_vertex, next_weight, g.num_vertices);
					prev[next_vertex] = current_vertex;
				}
			}
		}

		//Store current into source (which is prev)!!
		if(size < 1){
			printf("IT IS IMPOSSIBLE TO REACH DESTINATION\n");
			return fullpath;
		}
	}

	cost = current_weight;

	while(current_vertex != src_vertex){
		fullpath.push_back(g.vertex_to_name.at(current_vertex));
		current_vertex = prev[current_vertex];
	}
	fullpath.push_back(g.vertex_to_name.at(current_vertex));

	return fullpath;
}
