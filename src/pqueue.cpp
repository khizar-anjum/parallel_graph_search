#include "pqueue.h"
#include <cstdio>

void insert(int* heap, int &size, int item, int priority, int num_vertices){
	int parent = 0;
	int count = size;
	int temp;

	//increase the size of the heap and insert the stuffs there
	heap[size] = priority;
	heap[num_vertices + size] = item;
	size++;
	

	//now we will move on to correctly positioning the item
	while(1){
		parent = (count - 1)/2;

		//We compare if the parent has a smaller value else we replace
		if(parent > -1){
			if(heap[parent] > heap[count]){
				// swap the priorities
				temp = heap[count];
				heap[count] = heap[parent];
				heap[parent] = temp;

				// swap the items
				temp = heap[count + num_vertices];
				heap[count + num_vertices] = heap[parent + num_vertices];
				heap[parent + num_vertices] = temp;
			}
			else return;
		}
		else{
			return;
		}

		count = parent;
	}
}

void remove(int* heap, int &size, int index, int num_vertices){
	if(size < 1) return;
	int parent = 0;
	int leftchild = 0;
	int count = 0;
	int temp;

	//Decrease the size and replace the node with the last one!
	size--;
	//swap the priorities
	temp = heap[size];
	heap[size] = heap[index];
	heap[index] = temp;
	//swap the items
	temp = heap[size + num_vertices];
	heap[size + num_vertices] = heap[index + num_vertices];
	heap[index + num_vertices] = temp;
	
	//Since the element has been removed by just decreasing the size
	//we now simply percolate down!
	while(1){
		leftchild = (2*count) + 1;

		//Now we check if the right child exists!
		if(leftchild + 1 < size){
			if(heap[leftchild] > heap[leftchild + 1]){
				leftchild++;
			}
		}

		//Now, we have the one which is smaller in the leftchild
		if(leftchild < size){
			if(heap[count] > heap[leftchild]){
				//swap the priorities
				temp = heap[count];
				heap[count] = heap[leftchild];
				heap[leftchild] = temp;

				//swap the items
				temp = heap[count + num_vertices];
				heap[count + num_vertices] = heap[leftchild + num_vertices];
				heap[leftchild + num_vertices] = temp;			
			}
			else return;
		}
		else return;

		count = leftchild;
	}
}

void makeHeap(int* heap, int&size, int num_vertices){
	int count = size - 1;
	bool isleft = false;
	int subcount = 0;
	int temp;
	int leftchild = 0;
	int parent = 0;

	//If you have only one element in heap! its already sorted
	if(size <= 1) return;

	//making a heap in the first place

	//Exclude the possiblity that you only have a left child at the last index
	parent = (count-1)/2;
	leftchild = (2*count) + 1;
	if(size % 2 == 0){
		if(heap[count] < heap[parent]){
			//swap the priorities
			temp = heap[count];
			heap[count] = heap[parent];
			heap[parent] = temp;

			//swap the items
			temp = heap[count + num_vertices];
			heap[count + num_vertices] = heap[parent + num_vertices];
			heap[parent + num_vertices] = temp;		
		}
		count--;
	}

	while(1){
		isleft = false;
		parent = (count-1)/2;
		leftchild = (2*count) + 1;

		//Check which one is smaller! left or right
		if(count > 0){
			if(heap[count] > heap[count - 1]){
				isleft = true;
				count--;
			}
		}
		
		//Which ever is smaller, has now index at the count!
		if(heap[count] < heap[parent]){
			// swap the priorities
			temp = heap[count];
			heap[count] = heap[parent];
			heap[parent] = temp;

			//swap the items
			temp = heap[count + num_vertices];
			heap[count + num_vertices] = heap[parent + num_vertices];
			heap[parent + num_vertices] = temp;

			//Checking if the node has children which need some adjustment, who in turn have children to be checked
			subcount = count;
			while(1){
				leftchild = (2*subcount) + 1;
				//If both of the children are present
				if(leftchild + 1 < size){
					if(heap[leftchild] > heap[leftchild + 1]){
						leftchild++;
					}
					//Which ever is smaller is now at the leftchild (not necessarily leftchild now)
					//If child is smaller! swap it!
					if(heap[subcount] > heap[leftchild]){
						//swap the priorities
						temp = heap[subcount];
						heap[subcount] = heap[leftchild];
						heap[leftchild] = temp;

						//swap the items
						temp = heap[subcount + num_vertices];
						heap[subcount + num_vertices] = heap[leftchild + num_vertices];
						heap[leftchild + num_vertices] = temp;		

						//now moving to the next greater child!
						subcount = leftchild;
					}
					//else, may be we should end this, since all the other entries are already a heap
					else{
						break;
					}
				}	
				//If only the left child is present
				else if(leftchild < size){
					if(heap[subcount] > heap[leftchild]){
						//swap the priorities
						temp = heap[subcount];
						heap[subcount] = heap[leftchild];
						heap[leftchild] = temp;

						//swap the items
						temp = heap[subcount + num_vertices];
						heap[subcount + num_vertices] = heap[leftchild + num_vertices];
						heap[leftchild + num_vertices] = temp;		
					}

					//by heap shape property, there should be no children present for this!
					break;
				}
				//If nothing is present
				else{
					break;
				}
			}
		}

		if(isleft){
			count--;
		}
		else{
			count -= 2;
		}

		if(count < 0) break;
	}
}

void ExtractMin(int* heap, int &size, int& item, int& priority, int num_vertices){
	if(size > 0){
		priority = heap[0];
		item = heap[num_vertices];
		remove(heap, size, 0, num_vertices);
	}
}

void print_queue(int* heap, int &size, int num_vertices){
	for(int j = 0; j < size; j++)
		printf("%d->%d ", heap[j], heap[j+num_vertices]);
	printf("\n");
}

void getItem(int* heap, int size, int item, int& index, int& priority, int num_vertices){
	for(int j = 0; j < size; j++){
		if(heap[num_vertices + j] == item){
			index = j;
			priority = heap[j];
			return;
		}
	}
	index = -1;
	return;
}
