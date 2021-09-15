#include "Graph.h"
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;

Graph::Graph() {
	matrix = NULL;
	n = 0;
	nbEdges = 0;
}

Graph::Graph(int m) {
	matrix = NULL;
	resize(m);
}

int* Graph::operator[](int index) {
	if (index < 0 || index >= this->n) {
		cerr << "First node index out of range: " << index << "\n";
		matrix[-1] = 0; //Make it crash.
	}
	return matrix + this->n * index;
}

void Graph::resize(int m) {
	if (matrix != NULL) {
		delete[] matrix;
	}
	if (m > 0) {
		n = m;
		nbEdges = 0;
		matrix = new int[m * m];
		for (int i = 0; i < m * m; i++) {
			matrix[i] = 0;
		}
	}
}

Graph::~Graph() {
	resize(0);
}
