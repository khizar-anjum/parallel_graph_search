class graph{
public:
	int* index_arr; //Index Array
	int* num_connected; // Number of connected elements
	int* weight_arr; // Weight Array
	int* connected_to; // Connected To Array
	int num_vertices; //number of vertices
	int num_edges; //number of edges

	// Methods of the class
	graph(int num_vertices, int num_edges);
	~graph();
	int read(char* filename);
	int* get_connected_elements(int vertex);
	int* get_connected_weights(int vertex);
};

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
