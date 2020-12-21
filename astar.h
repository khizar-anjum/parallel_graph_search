#ifndef ASTAR__H
#define ASTAR__H

#include "graph.h"
#include "pqueue.h"

int* astar_seq(graph &g, int src, int (*h)(int));
std::vector<int> astar_seq(graph &g, int src, int dst, int &cost, int (*h)(int));
std::vector<int> dijkstra_seq(graph &g, int src, int dst, int &cost);
int* dijkstra_seq(graph &g, int src);

#endif