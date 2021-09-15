#ifndef MAKESOLUTION_H
#define MAKESOLUTION_H

#include "Graph.h"
#include <vector>
#include <iostream>
#include <limits.h>
#include <math.h>

using namespace std;

bool buildSolution(Graph& g,
	vector< vector<int> >& sol,
	vector< vector<int> >& neighbours,
	vector<int>& degree,
	vector< vector<double> >& t,
	int k, double alpha, double beta, int numISets,
	vector<int>& X, vector<int>& Y, vector< vector<int> >& tempX, vector< vector<int> >& tempY, vector< vector<int> >& ISet, vector<double>& tauEta);

void prettyPrintSolution(vector< vector<int> >& candSol);
void checkSolution(vector< vector<int> >& candSol, Graph& g);

#endif
