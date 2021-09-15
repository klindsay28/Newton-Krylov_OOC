#include "Kempe.h"
#include <limits.h>

extern unsigned long long numConfChecks;

//CONSTANTS USED FOR IDENTIFYING KEMP CHAINS
const int WHITE = 0;
const int GREY = 1;
const int BLACK = 2;

inline
void swap(int& a, int& b) {
	int temp; temp = a; a = b; b = temp;
}

//-------------------------------------------------------------------------------------
inline
int swapNodesInKempeChain(vector<vector<int> >& theColours, vector< vector<int> >& sol, vector< vector<int> >& theGroups, int group0, int group1)
{
	//Having Identified the Kempe chain, this operator now swaps the colors of the relevant nodes in
	//the two colour classes. It does this by making 2 new vectors and replacing the old ones
	int i, blackCnt = 0;
	vector<int> temp0, temp1;
	for (i = 0; i < theGroups[0].size(); i++) {
		if (theColours[0][i] == BLACK) {
			temp1.push_back(theGroups[0][i]);
			blackCnt++;
		}
		else temp0.push_back(theGroups[0][i]);
	}
	for (i = 0; i < theGroups[1].size(); i++) {
		if (theColours[1][i] == BLACK) {
			temp0.push_back(theGroups[1][i]);
			blackCnt++;
		}
		else temp1.push_back(theGroups[1][i]);
	}
	//Now replace the relavant parts of the candidate solution with these new vectors
	sol[group0].swap(temp0);
	sol[group1].swap(temp1);
	return blackCnt;
}

//-------------------------------------------------------------------------------------
inline
void DFSVisit(int uPos, int theGroup, Graph& g, vector< vector<int> >& theColours, vector< vector<int> >& theGroups)
{
	int theOtherGroup, vPos;
	if (theGroup == 0) theOtherGroup = 1;
	else theOtherGroup = 0;

	theColours[theGroup][uPos] = GREY;
	for (vPos = 0; vPos < theColours[theOtherGroup].size(); vPos++) {
		numConfChecks++;
		if (g[theGroups[theGroup][uPos]][theGroups[theOtherGroup][vPos]]) {
			if (theColours[theOtherGroup][vPos] == WHITE) {
				DFSVisit(vPos, theOtherGroup, g, theColours, theGroups);
			}
		}
	}
	theColours[theGroup][uPos] = BLACK;
}

//-------------------------------------------------------------------------------------
inline
int performKempeChainInterchange(vector< vector<int> >& sol, Graph& g, int group0, int group1, int nodePos)
{
	int i, blackCnt;
	//Set up some data structures for use with the restricted DepthFirstSearch Routine I am about to do
	vector< vector<int> > theColours(2);
	vector< vector<int> > theGroups(2);
	for (i = 0; i < sol[group0].size(); i++) {
		theColours[0].push_back(WHITE);
		theGroups[0].push_back(sol[group0][i]);
	}
	for (i = 0; i < sol[group1].size(); i++) {
		theColours[1].push_back(WHITE);
		theGroups[1].push_back(sol[group1][i]);
	}

	//Now start the restricted DFS from the chosen node
	DFSVisit(nodePos, 0, g, theColours, theGroups);

	//When we get to here we should be able to identify which nodes are in the Kempe chain by seeing which nodes are BLACK.
	//(note it could be that all nodes are black ==> colour relabelling. This serves no purpose, but oit is carried out anyway in this case.
	blackCnt = swapNodesInKempeChain(theColours, sol, theGroups, group0, group1);
	return blackCnt;
}
//-------------------------------------------------------------------------------------
void doRandomPeturbation(vector<int>& osp, int k, Graph& g)
{
	vector<vector<int> > sol(k, vector<int>());
	int i, j, kempeSize;
	int nodePos, group0, group1, groupSize;

	//Procedure that randomly alters a partial, proper g.col solution. First convert to parition form
	for (i = 0; i < g.n; i++) {
		if (osp[i] != INT_MIN) sol[osp[i] - 1].push_back(i);
	}

	//Now do the peturbation
	for (i = 0; i < k * 2; i++) {
		//Do a random Kempe chain move. First select a random group and a random vertex within it
		do {
			group0 = rand() % sol.size();
		} while (sol[group0].empty());
		nodePos = rand() % sol[group0].size();
		//And select a second group that is different to the first
		do {
			group1 = rand() % sol.size();
		} while (group0 == group1);
		groupSize = sol[group0].size() + sol[group1].size();
		kempeSize = performKempeChainInterchange(sol, g, group0, group1, nodePos);
		//(NOTE: if kempeSize==groupSize then this is just a colour relabelling)
	}

	//Now convert back to object form
	for (i = 0; i < g.n; i++) osp[i] = INT_MIN;
	for (i = 0; i < k; i++) {
		for (j = 0; j < sol[i].size(); j++) osp[sol[i][j]] = i + 1;
	}
}
