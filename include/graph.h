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

#endif