class graph{
public:
	int* index_arr; //Index Array
	int* num_connected; // Number of connected elements
	int* weight_arr; // Weight Array
	int* connected_to; // Connected To Array
	int num_vertices; //number of vertices
	int num_edges; //number of edges

	// Methods of the class
	graph();
	~graph();
	int read(char* filename);
	int* get_connected_elements(int vertex);
	int* get_connected_weights(int vertex);
}