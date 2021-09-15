#ifndef XOVER_INCLUDED
#define XOVER_INCLUDED

#include "Graph.h"
#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <vector>

using namespace std;

void doCrossover(int xOverType, vector<int>& osp, vector<int>& parents, Graph& g, int k, vector<vector<int> >& population);

#endif
