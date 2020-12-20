#include "graph.h"

void astar(graph &g, int src){
	printf("%d\n", src);
}


void dijkstra(graph &g, int src, int dst){
	printf("%d %d\n", src, dst);
}
/*
vector<string> dijkstra(string city, string destination, float &d){
	//THE MASLA IS THAT IT IS COMPARING POINTER VALUES INSTEAD OF IT COMPARING DISTANCES!
	node* source = NULL;
	node* dest = NULL;
	node* current;
	vector<string> fullpath;
	d = 0.0;


	//initialize all the values with +inf, visit being zero and previous being NULL!
	for(int i = 0; i < cities.size(); i++){
		cities[i]->dist = 0x7f800000;
		cities[i]->visit = 0;
		cities[i]->prev = NULL;
		if(cities[i]->name == city){ 
			source = cities[i];
			source->dist = 0.0;
		}
		if(cities[i]->name == destination) dest = cities[i];
	}

	//Check if cities are even present or not!
	if(!source || !dest){
		cout << "ERROR! Source or Destination city is not present in the graph!" << endl;
		return fullpath;
	}

	//making a priority queue from the vertices here! And defining sets!
	ppqueue<node*> myqueue(cities);
	set<node*> nodeset;
	set<node*>::iterator it1;
	source = NULL;

	//Now, extracting mins and carrying on till we find the shit!
	while(1){
		//Extracting min and pulling the corresponding vertex into cloud
		current = myqueue.ExtractMin();
		nodeset.insert(current);
	//	current->prev = source;
	//	cout << "current is " << current->name << " its dist is " << current->dist << endl;
		//Destination found mechanism!
		if(current->name == destination){
			break;
		}

		//Run through every edge and update the distances!
		for(int i = 0; i < current->edges.size(); i++){
			if(current->edges[i].Origin == current){
				it1 = nodeset.find(current->edges[i].Dest);
				if(it1 == nodeset.end()){
					if(current->edges[i].Dest->dist > current->dist + current->edges[i].weight){
						current->edges[i].Dest->dist = current->dist + current->edges[i].weight;
						current->edges[i].Dest->prev = current;
					}
				}
			}
			else{
				it1 = nodeset.find(current->edges[i].Origin);
				if(it1 == nodeset.end()){
					if(current->edges[i].Origin->dist > current->dist + current->edges[i].weight){
						current->edges[i].Origin->dist = current->dist + current->edges[i].weight;
						current->edges[i].Origin->prev = current;
					}
				}
			}
		}

		//make heap again, because we have changed the values!
		myqueue.makeHeap();

		//Store current into source (which is prev)!!
		source = current;
	}

	d = current->dist;

	while(current != NULL){
		fullpath.push_back(current->name);
		current = current->prev;
	}

	return fullpath;
}
*/