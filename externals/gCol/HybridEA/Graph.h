#ifndef GraphIncluded
#define GraphIncluded

class Graph {

public:

	Graph();
	Graph(int n);
	~Graph();

	void resize(int n);

	unsigned char** matrix;
	int n;        // number of nodes
	int nbEdges;  // number of edges

	unsigned char* operator[](int index);

};

#endif
