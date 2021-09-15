#ifndef TABU_INCLUDED
#define TABU_INCLUDED

#include <vector>
#include "Graph.h"

using namespace std;

int tabu(Graph& g, vector<int>& c, int k, int maxIterations, int verbose, int** neighbors);

#endif
