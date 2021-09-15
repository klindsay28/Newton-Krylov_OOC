#ifndef INITIALIZECOLORING_INCLUDED
#define INITIALIZECOLORING_INCLUDED

#include "Graph.h"
#include <vector>

int generateInitialK(Graph& g, int alg, int* bestColouring);
void initializeColoring(Graph& g, int* c, int k);
void initializeColoringForTabu(Graph& g, int* c, int k);

#endif
