#ifndef PQUEUE__CUH
#define PQUEUE__CUH

/*
	THIS FILE PROVIDES FUNCTIONS TO IMPLEMENT A PRIORITY_QUEUE
	this file works on an int array which is supposed to be heapified
	These functions work by-reference on two main variables:
		1. int* heap
		2. int size

	the total size of heap should be 2*num_vertices meaning first num_vertices 
	space is reserved for priority values and the rest is reserved for storing
	names of vertices

	the size vector can only have a max value of num_vertices and it tells us 
	the current number of entries in the queue ispite of its allocated size

	I did not create a class for this to facilitate parallelization in CUDA
*/

__device__ void ExtractMin_GPU(int* heap, int &size, int& item, int& priority, int num_vertices);		
__device__ void insert_GPU(int* heap, int &size, int item, int priority, int num_vertices);
__device__ void remove_GPU(int* heap, int &size, int index, int num_vertices);
__device__ void getItem_GPU(int* heap, int size, int item, int& index, int& priority, int num_vertices);
__device__ void makeHeap_GPU(int* heap, int&size, int num_vertices);

#endif