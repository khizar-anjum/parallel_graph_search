#include "astar.h"
#include "pqueue.h"
#include "bellmanford.h"
#include "astar.cuh"
#include "bellmanford.cuh"
#include "pqueue.cuh"
#include <cstdio>
#include <iostream>
#include <chrono>

int heuristic(int a){
	return 0;
}

int main(){
	std::cout << "Welcome to Parallel Graph Search" << std::endl;
	/*
	std::cout << "Enter Name of File: ";
	char buf[256];
	std::cin >> buf;
	graph g(buf);
	*/
	graph g("data/USA-road-d.NY.csv");
	//graph g("data/socdata_new.csv");
	int* heap = new int[2*g.num_vertices];
	int size = 0;
	int total_cost = 0;
	int* costs;
	int time = 0;
	std::chrono::steady_clock::time_point begin_exec;
	std::chrono::steady_clock::time_point end_exec;
	FILE* pfile = fopen(("results/results_" + std::to_string(g.num_vertices) + ".txt").c_str(), "w");
	std::cout << "Read the file into memory. Starting tests.." << std::endl << std::endl;
	
	begin_exec = std::chrono::steady_clock::now();
	costs = astar_seq(g, 1, heuristic);
	end_exec = std::chrono::steady_clock::now();
	time = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end_exec - begin_exec).count());
	printf("For Astar Sequential it took a total of %d microseconds\n", time);
	fprintf(pfile, "%d\n", time);

	begin_exec = std::chrono::steady_clock::now();
	costs = bellmanford_seq(g, 1);
	end_exec = std::chrono::steady_clock::now();
	time = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end_exec - begin_exec).count());
	printf("For Bellmanford Sequential it took a total of %d microseconds\n", time);
	fprintf(pfile, "%d\n", time);
	printf("\n");
	
	int numqs[5] = {500, 1000, 2000, 5000, 10000};
	for(int i = 0; i < 5; i++){
		begin_exec = std::chrono::steady_clock::now();
		costs = astar_par(g, 1, numqs[i]);
		end_exec = std::chrono::steady_clock::now();
		time = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end_exec - begin_exec).count());
		printf("For numqs %d, it took a total of %d microseconds\n", numqs[i], time);
		fprintf(pfile, "%d %d\n", numqs[i], time);
	}
	printf("\n");

	int blocksize[3] = {512, 768, 1024};
	int gridsize[3] = {256, 512, 1024};
	for(int j = 0; j < 3; j++){
		for(int i = 0; i < 3; i++){
			begin_exec = std::chrono::steady_clock::now();
			costs = bellmanford_par(g, 1, blocksize[i], gridsize[j]);
			end_exec = std::chrono::steady_clock::now();
			time = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end_exec - begin_exec).count());
			printf("For blocksize %d and gridsize %d, it took a total of %d microseconds\n", blocksize[i], gridsize[j], time);
			fprintf(pfile, "%d %d %d\n", blocksize[i], gridsize[j], time);
		}
		printf("\n");
	}
	fclose(pfile);
	return 0;
}

