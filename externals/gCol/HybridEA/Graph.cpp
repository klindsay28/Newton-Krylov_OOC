
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

unsigned char* Graph::operator[](int index) {
	if (index < 0 || index >= this->n) {
		cerr << "First node index out of range: " << index << "\n";
		matrix[-1] = 0; //Make it crash.
	}
	return matrix[index];
}

void Graph::resize(int m) {
	if (matrix != NULL) {
		for (int i = 0; i < n; i++) {
			delete[] matrix[i];
		}
		delete[] matrix;
	}
	if (m > 0) {
		n = m;
		nbEdges = 0;
		matrix = new unsigned char* [m];
		for (int i = 0; i < m; i++) {
			matrix[i] = new unsigned char[m];
			for (int j = 0; j < m; j++) {
				matrix[i][j] = 0;
			}
		}
	}
}


Graph::~Graph() {
	resize(0);
}
