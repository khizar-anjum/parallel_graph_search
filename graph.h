#ifndef GRAPH__H
#define GRAPH__H

#include <map>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>

//THIS FILE PROVIDES FUNCTIONS TO IMPLEMENT A GRAPH CLASS

class graph{
public:
	int* index_arr; //Index Array
	int* num_connected; // Number of connected elements
	int* weight_arr; // Weight Array
	int* connected_to; // Connected To Array
	int num_vertices; //number of vertices
	int num_edges; //number of edges

	//maps from and to vertex names
	std::map<int, int> name_to_vertex;
	std::map<int, int> vertex_to_name;

	//constructors and destructors
	graph(std::string filename);
	graph(int num_vertices, int num_edges);
	~graph();

	// Methods of the class
	void print_connected_elements(int vertex);
	void print_connected_weights(int vertex);
};

graph::graph(std::string filename){
	// Expects a csv file format with edges describes in 3 column format
	// 3 columns being SOURCE, TARGET, WEIGHT
	// All the values should be ints

	int i = 0;
 	int num_cols = 3;
	std::ifstream inputFile(filename);
 	std::string line; 
 	std::vector<int> from;
 	std::vector<int> to;
 	std::vector<int> weights;
 	std::vector<int> all_vertices;

	while(std::getline(inputFile, line)){
		std::istringstream ss(line);
		while(getline(ss, line, ',')){
			// using comma as a separator as its a csv file format
			if(i == 0){ // its the SOURCE column
				from.push_back(std::stoi(line));
			}
			else if(i == 1){ // its the TARGET column
				to.push_back(std::stoi(line));
			}
			else{ // its the WEIGHT column
				weights.push_back(std::stoi(line));
			}
			i = ++i % num_cols;
		}
	}
	inputFile.close();

	// concatenate to get all the vertices
	// thanks to https://stackoverflow.com/questions/3177241/what-is-the-best-way-to-concatenate-two-vectors/3177254
	all_vertices.reserve(from.size() + to.size());
	all_vertices.insert(all_vertices.end(), from.begin(), from.end());
	all_vertices.insert(all_vertices.end(), to.begin(), to.end());

	//lets remove duplicates
	//thanks to https://stackoverflow.com/questions/1041620/whats-the-most-efficient-way-to-erase-duplicates-and-sort-a-vector
	std::sort(all_vertices.begin(), all_vertices.end());
	all_vertices.erase(std::unique(all_vertices.begin(), all_vertices.end()), all_vertices.end());

	//lets initialize the arrays
	num_vertices = all_vertices.size();
	num_edges = weights.size();
	index_arr = new int[num_vertices];
	num_connected = new int[num_vertices];
	weight_arr = new int[num_edges];
	connected_to = new int[num_edges];

	//lets initialize the maps and the index and num_connected arrays
	i = 0;
	for(int j = 0; j < all_vertices.size(); j++){
		name_to_vertex.insert(std::pair<int, int>(all_vertices[j], j));
		vertex_to_name.insert(std::pair<int, int>(j, all_vertices[j]));

		index_arr[j] = i;
		num_connected[j] = std::count(from.begin(), from.end(), all_vertices[j]);
		i += num_connected[j];
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