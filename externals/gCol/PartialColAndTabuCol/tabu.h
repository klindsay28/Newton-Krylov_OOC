#ifndef TABU_INCLUDED
#define TABU_INCLUDED

#include "Graph.h"

long tabu(Graph& g, int* c, int k, unsigned long long maxIterations, int tenure, int verbose, int frequency, int increment, int** neighbors);


#endif
