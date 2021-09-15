#ifndef INITSOLGEN_H
#define INITSOLGEN_H

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <list>
#include <math.h>
#include "hillClimber.h"

void makeInitSolution(int numNodes, vector< vector<int> >& candSol, vector< vector<int> >& tempSol, vector<int>& colNode);

void DSaturCol(vector< vector<int> >& candSol, int numNodes, vector<int>& colNode);

bool colourIsFeasible(int v, vector< vector<int> >& sol, int c, vector<int>& colNode);

void assignAColourDSatur(bool& foundColour, vector< vector<int> >& candSol, vector<int>& permutation, int nodePos, vector<int>& satDeg,
	int numNodes, vector<int>& colNode);

void swap(int& a, int& b);

#endif //INITSOLGEN_H
