#ifndef DIJKSTRA__H
#define DIJKSTRA__H

#include "graph.h"
#include "pqueue.h"

std::vector<int> dijkstra_seq(graph &g, int src, int dst, int &cost);
int* dijkstra_seq(graph &g, int src);

#endif