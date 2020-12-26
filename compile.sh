nvcc pqueue.o astar.cu bellmanford.cu dijkstra.cu bellmanford.cpp astar.cpp dijkstra.cpp pqueue.cpp main.cpp graph.cpp --device-c -arch sm_75 -o pg_search
