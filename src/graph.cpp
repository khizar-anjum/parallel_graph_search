#ifndef GRAPH__CPP
#define GRAPH__CPP

#include "graph.h"
#include "rapidcsv.h"

graph::graph(std::string filename){
	// Expects a csv file format with edges describes in 3 column format
	// 3 columns being SOURCE, TARGET, WEIGHT
	// All the values should be ints
 	std::vector<int> all_vertices;

 	rapidcsv::Document doc(filename, rapidcsv::LabelParams(-1, -1));

 	std::vector<int> from = doc.GetColumn<int>(0);
 	std::vector<int> to = doc.GetColumn<int>(1);
 	std::vector<int> weights = doc.GetColumn<int>(2);
 	
	// concatenate to get all the vertices
	// thanks to https://stackoverflow.com/questions/3177241/what-is-the-best-way-to-concatenate-two-vectors/3177254
	all_vertices.reserve(from.size() + to.size());
	all_vertices.insert(all_vertices.end(), from.begin(), from.end());
	all_vertices.insert(all_vertices.end(), to.begin(), to.end());
	
	//lets remove duplicates
	//thanks to https://stackoverflow.com/questions/1041620/whats-the-most-efficient-way-to-erase-duplicates-and-sort-a-vector
	std::sort(all_vertices.begin(), all_vertices.end());
	all_vertices.erase(std::unique(all_vertices.begin(), all_vertices.end()), all_vertices.end());
	//sort the from vector
	std::vector<int> from_sorted = from;
	std::sort(from_sorted.begin(), from_sorted.end());
	
	//lets initialize the arrays
	num_vertices = all_vertices.size();
	num_edges = weights.size();
	index_arr = new int[num_vertices];
	num_connected = new int[num_vertices];
	weight_arr = new int[num_edges];
	connected_to = new int[num_edges];

	//lets initialize the maps and the index and num_connected arrays
	int i = 0;
	for(int j = 0; j < all_vertices.size(); j++){
		name_to_vertex.insert(std::pair<int, int>(all_vertices[j], j));
		vertex_to_name.insert(std::pair<int, int>(j, all_vertices[j]));

		index_arr[j] = i;
		num_connected[j] = 0;
		while(true){
			num_connected[j]++;
			if(i < from.size() && from_sorted[i] != from_sorted[i+1]){
				break;
			}
			i++;
		}
		i++;
	}
	//lets add values to the weights and connected to arrays
	int localiterator[num_vertices] = {0};
	int curr_index = 0;
	for(int j = 0; j < from.size(); j++){
		curr_index = index_arr[name_to_vertex.at(from[j])] + localiterator[name_to_vertex.at(from[j])];
		weight_arr[curr_index] = weights[j];
		connected_to[curr_index] = name_to_vertex.at(to[j]);
		localiterator[name_to_vertex.at(from[j])]++;
	}
}

graph::graph(int num_vertices_, int num_edges_){
	num_vertices = num_vertices_;
	num_edges = num_edges_;
	index_arr = new int[num_vertices];
	num_connected = new int[num_vertices];
	weight_arr = new int[num_edges];
	connected_to = new int[num_edges];
}

graph::~graph(){
	delete [] index_arr;
	delete [] num_connected;
	delete [] weight_arr;
	delete [] connected_to;
}

void graph::print_connected_elements(int vertex){
	std::map<int, int>::iterator it = name_to_vertex.find(vertex);
	if(it == name_to_vertex.end()) throw std::invalid_argument( "Invalid vertex name" );
	int inkey = it->second;
	for(int j = 0; j < num_connected[inkey]; j++){
		printf("%d ", vertex_to_name.at(connected_to[index_arr[inkey]+j]));
	}
	printf("\n");
}

void graph::print_connected_weights(int vertex){
	std::map<int, int>::iterator it = name_to_vertex.find(vertex);
	if(it == name_to_vertex.end()) throw std::invalid_argument( "Invalid vertex name" );
	int inkey = it->second;
	for(int j = 0; j < num_connected[inkey]; j++){
		printf("%d ", weight_arr[index_arr[inkey]+j]);
	}
	printf("\n");
}
#endif