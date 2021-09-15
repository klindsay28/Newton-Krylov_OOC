#ifndef DISPLAY_H
#define DISPLAY_H

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <list>
#include <math.h>
#include "hillClimber.h"

void prettyPrintSolution(vector< vector<int> >& candSol);
//Just prints the solution to the screen

void checkSolution(vector< vector<int> >& candSol, int numItems);
//Takes a complete solution and checks that it represents a feasible colouring.
//Returns error messages if the colouring is wrong/illegal

void readInputFile(ifstream& inStream, int& numNodes, int& numEdges);

#endif //DISPLAY_H
