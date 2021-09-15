#ifndef MANIPULATEARRAYS_INCLUDED
#define MANIPULATEARRAYS_INCLUDED

#include "Graph.h"

void initializeArrays(int**& nodesByColor, int**& conflicts,
	int**& tabuStatus, int*& nbcPosition, Graph& g, int* c, int k);

void freeArrays(int**& nodesByColor, int**& conflicts, int**& tabuStatus,
	int*& nbcPosition, int k, int n);

void moveNodeToColorForTabu(int bestNode, int bestColor, Graph& g, int* c,
	int** nodesByColor, int** conflicts,
	int* nbcPosition, int** neighbors,
	int* nodesInConflict, int* confPosition,
	int** tabuStatus, long totalIterations, int tabuTenure);

#endif
