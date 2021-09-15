#ifndef TABU_INCLUDED
#define TABU_INCLUDED

#include "Graph.h"

int tabu(Graph& g, int** neighbors, int* c, int k, int maxIterations, int verbose);

#endif
