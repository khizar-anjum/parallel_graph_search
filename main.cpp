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
	/*
	begin_exec = std::chrono::steady_clock::now();
	costs = astar_seq(g, 1, heuristic);
	end_exec = std::chrono::steady_clock::now();
	printf("For Astar Sequential it took a total of %d milliseconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_exec - begin_exec).count());

	begin_exec = std::chrono::steady_clock::now();
	costs = astar_par(g, 1);
	end_exec = std::chrono::steady_clock::now();
	printf("For Astar Parallel it took a total of %d milliseconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_exec - begin_exec).count());

	begin_exec = std::chrono::steady_clock::now();
	costs = bellmanford_seq(g, 1);
	end_exec = std::chrono::steady_clock::now();
	printf("For Bellmanford Sequential it took a total of %d milliseconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_exec - begin_exec).count());

	begin_exec = std::chrono::steady_clock::now();
	costs = dijkstra_seq(g, 1);
	end_exec = std::chrono::steady_clock::now();
	printf("For Bellmanford Parallel it took a total of %d milliseconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_exec - begin_exec).count());
	
	int numqs[6] = {10, 50, 100, 500, 1000, 2000};
	for(int i = 0; i < 6; i++){
		begin_exec = std::chrono::steady_clock::now();
		costs = astar_par(g, 1, numqs[i]);
		end_exec = std::chrono::steady_clock::now();
		printf("For numqs %d, it took a total of %d milliseconds\n", numqs[i], std::chrono::duration_cast<std::chrono::milliseconds>(end_exec - begin_exec).count());
	}
	*/

	int grids[6] = {64, 128, 256, 512, 768, 1024};
	for(int i = 0; i < 6; i++){
		begin_exec = std::chrono::steady_clock::now();
		costs = bellmanford_par(g, 1, grids[i],64);
		end_exec = std::chrono::steady_clock::now();
		printf("For grids %d, it took a total of %d milliseconds\n", grids[i], std::chrono::duration_cast<std::chrono::milliseconds>(end_exec - begin_exec).count());
	}
	return 0;
}

