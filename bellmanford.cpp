#include "bellmanford.h"

int* bellmanford_seq(graph &g, int src){
	// starts at the source and calculates the distance for all the 
	// vertices inside the graph
	int* costs = new int[g.num_vertices];
	// lets initialize these costs to infinity (A high enough number)
	for(int i = 0; i < g.num_vertices; i++){
		costs[i] = 0x70000000;
	}

	// check if src exists and populate it
	std::map<int, int>::iterator it = g.name_to_vertex.find(src);
	if(it == g.name_to_vertex.end()) throw std::invalid_argument( "Invalid vertex name" );
	int src_vertex = it->second;

	costs[src_vertex] = 0;
	int connected_from = 0;
	int k = 1;

	//Repeating for |V| - 1 times 
	for(int j = 0; j < g.num_vertices - 1; j++){
		k = 1;
		connected_from = 0;
		//Run through every edge and update the distances!
		for(int i = 0; i < g.num_edges; i++){
			// using the fact that from_vertices are sorted!
			if(g.num_connected[connected_from] < k){
				connected_from++;
				k = 1;
			}
			//lets see if we can update this
			if(costs[g.connected_to[i]] > g.weight_arr[i] + costs[connected_from]){
				costs[g.connected_to[i]] = g.weight_arr[i] + costs[connected_from];
			}
			k++;
		}
	}
	
	k = 1;
	connected_from = 0;
	//Run through every edge and update the distances!
	for(int i = 0; i < g.num_edges; i++){
		// using the fact that from_vertices are sorted!
		if(g.num_connected[connected_from] < k){
			connected_from++;
			k = 1;
		}
		//lets see if we can update this
		if(costs[g.connected_to[i]] > g.weight_arr[i] + costs[connected_from]){
			throw std::logic_error( "Graph contains a negative-weight cycle" );
		}

		k++;
	}
	
	return costs;
}