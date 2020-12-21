#include "astar.h"
#include "pqueue.h"
#include "bellmanford.h"
#include "astar.cuh"
#include "bellmanford.cuh"
#include "pqueue.cuh"
#include <cstdio>
#include <chrono>

int heuristic(int a){
	return 0;
}

int main(){
	graph g("data/socdata_new.csv");
	//graph g("../soc-sign-bitcoinalpha.csv");
	int* heap = new int[2*g.num_vertices];
	printf("%d %d\n", g.num_vertices, g.num_edges);
	int size = 0;
	int total_cost = 0;
	int* costs;
	std::chrono::steady_clock::time_point begin_exec;
	std::chrono::steady_clock::time_point end_exec;

	begin_exec = std::chrono::steady_clock::now();
	costs = dijkstra_seq(g, 1);
	end_exec = std::chrono::steady_clock::now();
	printf("For Astar it took a total of %d milliseconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_exec - begin_exec).count());
	

	return 0;
}

