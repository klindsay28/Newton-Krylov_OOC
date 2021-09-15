#ifndef MAKESOLUTION_H
#define MAKESOLUTION_H

#include "Graph.h"
#include <vector>
#include <iostream>
#include <limits.h>
#include <string>
#include <stdlib.h>

using namespace std;

int generateInitialK(Graph& g, int alg, vector<int>& bestColouring);
void makeInitSolution(Graph& g, vector<int>& sol, int k, int verbose);

#endif
