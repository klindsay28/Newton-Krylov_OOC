#ifndef GraphIncluded
#define GraphIncluded

class Graph {

public:

	Graph();
	Graph(int n);
	~Graph();

	void resize(int n);

	int* matrix;
	int n;        // number of nodes
	int nbEdges;  // number of edges

	int* operator[](int index);

};

#endif
