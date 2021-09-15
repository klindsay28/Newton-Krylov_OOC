#ifndef ALGFNS_H
#define ALGFNS_H

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <list>
#include <math.h>
#include "hillClimber.h"

//----------------------------------------------------------------------------------------------------------------
//PERMUTING AND SORTING FUNCTIONS
void RandomlyPermuteASection(vector<int>& permutation, int left, int right);
void randomlyPermuteTheGroups(vector< vector<int> >& candSol);
void sortGroupsReverse(vector< vector<int> >& A);
void randomlyPermuteWithinEachGroup(vector< vector<int> >& candSol);

//----------------------------------------------------------------------------------------------------------------
//SEARCH FUNCTIONS
void applySearch(vector< vector<int> >& candSol, double removalRate, int lsLimit, vector< vector<int> >& newSol,
	vector<int>& groupsToRemove, vector< vector<int> >& unplaced, vector<int>& colNode);
//The first search routine. Basically removes one or more groups (dictated by remGroups)
//and then forms a list U of unplaced events and passes the incomplete grid and U inth the LS
//routine. When the LS is finished, if U is non empty then the constructive colouring procedure
//is called

void runLS(vector< vector<int> >& candSol, vector< vector<int> >& U, int itLimit);
//The s+f procedure - this is the main procedure of the algorithm

void rebuild(vector< vector<int> >& candSol, vector< vector<int> >& newSol, vector<int>& colNode);


//----------------------------------------------------------------------------------------------------------------
//NEIGHBOURHOOD FUNCTIONS
void performItemSwap(int& firstGroup, int& secGroup, vector< vector<int> >& candSol, vector<int>& randPerm, bool& doneAMove);

void performKempeChainInterchange(vector< vector<int> >& candSol, int& group0, int& group1,
	vector< vector<int> >& theColours, vector< vector<int> >& theGroups, vector<int>& temp0, vector<int>& temp1, bool& elimColour, bool& doneAMove);

void DFSVisit(int uPos, int theGroup, vector< vector<int> >& theColours, vector< vector<int> >
	& theGroups, int& blackCount);

void swapNodesInKempeChain(vector< vector<int> >& theColours, vector< vector<int> >& candSol, vector< vector<int> >& theGroups,
	int group0, int group1, vector<int>& temp0, vector<int>& temp1);

bool neighbourhoodSwapIsFeasible(vector< vector<int> >& candSol, int g1, int i1, int g2, int i2);

bool groupIsSuitable(int v, vector< vector<int> >& sol, int c, vector<int>& colNode);



//----------------------------------------------------------------------------------------------------------------
//THREE VERSIONS OF THE FORCE FUNCTION
void force(vector< vector<int> >& U, vector< vector<int> >& candSol);
void force(vector< vector<int> >& U, vector< vector<int> >& candSol, vector<int>& groupsToConsider);
//----------------------------------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------------------------------
//MISCILANNEOUS
void eraseElementFromIntVector(vector<int>& theVec, int thePos);

void selectGroupsToRemove(vector<int>& groupsToRemove, int currentNumGroups, double removalRate);
//This is the function that selects which groups to remove from the solution. A number of
//error checks are performed here to make sure all goes well

void removeAnyEmptyGroups(vector< vector<int> >& candSol);

void solDist(vector< vector<int> >& S, vector< vector<int> >& T);
//----------------------------------------------------------------------------------------------------------------



#endif //ALGFNS
