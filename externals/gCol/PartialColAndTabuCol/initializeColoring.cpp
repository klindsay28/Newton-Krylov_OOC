#include "initializeColoring.h"
#include <stdlib.h>
#include <limits.h>

using namespace std;

extern unsigned long long numConfChecks;

inline
void swap(int& a, int& b) {
	int temp;
	temp = a; a = b; b = temp;
}

inline
bool colourIsFeasible(int v, vector< vector<int> >& sol, int c, vector<int>& colNode, vector< vector<int> >& adjList, Graph& g)
{
	//Checks to see whether vertex v can be feasibly inserted into colour c in sol.
	int i;
	numConfChecks++;
	if (sol[c].size() > adjList[v].size()) {
		//check if any neighbours of v are currently in colour c
		for (i = 0; i < adjList[v].size(); i++) {
			numConfChecks++;
			if (colNode[adjList[v][i]] == c) return false;
		}
		return true;
	}
	else {
		//check if any vertices in colour c are adjacent to v
		for (i = 0; i < sol[c].size(); i++) {
			numConfChecks++;
			if (g[v][sol[c][i]]) return false;
		}
		return true;
	}
}

inline
void assignAColourDSatur(bool& foundColour, vector< vector<int> >& candSol, vector<int>& permutation, int nodePos, vector<int>& satDeg, Graph& g, vector<int>& colNode, vector< vector<int> >& adjList)
{
	int i, j, c = 0, v = permutation[nodePos];
	bool alreadyAdj;

	while (c < candSol.size() && !foundColour) {
		//check if colour c is feasible for vertex v
		if (colourIsFeasible(v, candSol, c, colNode, adjList, g)) {
			//v can be added to this colour
			foundColour = true;
			candSol[c].push_back(v);
			colNode[v] = c;
			//We now need to update satDeg. To do this we identify the uncloured nodes i that are adjacent to
			//this newly coloured node v. If i is already adjacent to a node in colour c we do nothing,
			//otherwise its saturation degree is increased...
			for (i = 0; i < satDeg.size(); i++) {
				numConfChecks++;
				if (g[v][permutation[i]]) {
					alreadyAdj = false;
					j = 0;
					while (j < candSol[c].size() - 1 && !alreadyAdj) {
						numConfChecks++;
						if (g[candSol[c][j]][permutation[i]]) alreadyAdj = true;
						j++;
					}
					if (!alreadyAdj)
						satDeg[i]++;
				}
			}
		}
		c++;
	}
}

inline
void greedyCol(vector< vector<int> >& candSol, vector<int>& colNode, Graph& g, vector< vector<int> >& adjList)
{
	//1) Make an empty vector representing all the unplaced nodes (i.e. all of them) and permute
	int i, r, j;
	vector<int> a(g.n);
	for (i = 0; i < g.n; i++) a[i] = i;
	for (i = g.n - 1; i >= 0; i--) {
		r = rand() % (i + 1);
		swap(a[i], a[r]);
	}

	//Now colour using the greedy algorithm. First, place the first node into the first group
	candSol.clear();
	candSol.push_back(vector<int>());
	candSol[0].push_back(a[0]);
	colNode[a[0]] = 0;

	//Now go through the remaining nodes and see if they are suitable for any existing colour. If it isn't, we create a new colour
	for (i = 1; i < g.n; i++) {
		for (j = 0; j < candSol.size(); j++) {
			if (colourIsFeasible(a[i], candSol, j, colNode, adjList, g)) {
				//the Item can be inserted into this group. So we do
				candSol[j].push_back(a[i]);
				colNode[a[i]] = j;
				break;
			}
		}
		if (j >= candSol.size()) {
			//If we are here then the item could not be inserted into any of the existing groups. So we make a new one
			candSol.push_back(vector<int>());
			candSol.back().push_back(a[i]);
			colNode[a[i]] = candSol.size() - 1;
		}
	}
}

inline
void DSaturCol(vector< vector<int> >& candSol, vector<int>& colNode, Graph& g, vector< vector<int> >& adjList)
{
	int i, j, r;
	bool foundColour;

	//Make a vector representing all the nodes
	vector<int> permutation(g.n);
	for (i = 0; i < g.n; i++)permutation[i] = i;
	//Randomly permute the nodes, and then arrange by increasing order of degree
	//(this allows more than 1 possible outcome from the sort procedure)
	for (i = permutation.size() - 1; i >= 0; i--) {
		r = rand() % (i + 1);
		swap(permutation[i], permutation[r]);
	}
	//Bubble sort is used here. This could be made more efficent
	for (i = (permutation.size() - 1); i >= 0; i--) {
		for (j = 1; j <= i; j++) {
			numConfChecks += 2;
			if (adjList[permutation[j - 1]].size() > adjList[permutation[j]].size()) {
				swap(permutation[j - 1], permutation[j]);
			}
		}
	}

	//We also have a vector to hold the saturation degrees of each node
	vector<int> satDeg(permutation.size(), 0);

	//Initialise candSol and colNode
	candSol.clear();
	candSol.push_back(vector<int>());
	for (i = 0; i < colNode.size(); i++) colNode[i] = INT_MIN;

	//Colour the rightmost node first (it has the highest degree), and remove it from the permutation
	candSol[0].push_back(permutation.back());
	colNode[permutation.back()] = 0;
	permutation.pop_back();
	//..and update the saturation degree array
	satDeg.pop_back();
	for (i = 0; i < satDeg.size(); i++) {
		numConfChecks++;
		if (g[candSol[0][0]][permutation[i]]) {
			satDeg[i]++;
		}
	}

	//Now colour the remaining nodes.
	int nodePos = 0, maxSat;
	while (!permutation.empty()) {
		//choose the node to colour next (the rightmost node that has maximal satDegree)
		maxSat = INT_MIN;
		for (i = 0; i < satDeg.size(); i++) {
			if (satDeg[i] >= maxSat) {
				maxSat = satDeg[i];
				nodePos = i;
			}
		}
		//now choose which colour to assign to the node
		foundColour = false;
		assignAColourDSatur(foundColour, candSol, permutation, nodePos, satDeg, g, colNode, adjList);
		if (!foundColour) {
			//If we are here we have to make a new colour as we have tried all the other ones and none are suitable
			candSol.push_back(vector<int>());
			candSol.back().push_back(permutation[nodePos]);
			colNode[permutation[nodePos]] = candSol.size() - 1;
			//Remember to update the saturation degree array
			for (i = 0; i < permutation.size(); i++) {
				numConfChecks++;
				if (g[permutation[nodePos]][permutation[i]]) {
					satDeg[i]++;
				}
			}
		}
		//Finally, we remove the node from the permutation
		permutation.erase(permutation.begin() + nodePos);
		satDeg.erase(satDeg.begin() + nodePos);
	}
}

int generateInitialK(Graph& g, int alg, int* bestColouring)
{
	//Produce an solution using a constructive algorithm to get an intial setting for k
	int i, j;

	//Make the structures needed for the constructive algorithms
	vector< vector<int> > candSol, adjList(g.n, vector<int>());
	vector<int> colNode(g.n, INT_MAX);
	for (i = 0; i < g.n; i++) {
		for (j = 0; j < g.n; j++) {
			if (g[i][j] && i != j) {
				adjList[i].push_back(j);
			}
		}
	}
	//Now make the solution
	if (alg == 1) DSaturCol(candSol, colNode, g, adjList);
	else greedyCol(candSol, colNode, g, adjList);
	//Copy this solution into bestColouring
	for (i = 0; i < candSol.size(); i++) for (j = 0; j < candSol[i].size(); j++) bestColouring[candSol[i][j]] = i;
	//And return the number of colours it has used
	return candSol.size();
}

void initializeColoring(Graph& g, int* c, int k)
{
	// A simple greedy algorithm that leaves the assigned color if possible, gives another legal color or
	// assigns color 0 if nothing else is available.
	// First produce a random permutation for the vertex order
	int* perm = new int[g.n];
	for (int i = 0; i < g.n; i++) {
		perm[i] = i;
	}
	for (int i = 0; i < g.n; i++) {
		int p = rand() % g.n;
		int h = perm[i];
		perm[i] = perm[p];
		perm[p] = h;
	}

	int* taken = new int[k + 1];

	// Insure all colors are in the range [0, ... ,k]
	for (int i = 0; i < g.n; i++) {
		if (c[i]<0 || c[i]>k) c[i] = 0;
	}

	// Go through all nodes
	for (int ii = 0; ii < g.n; ii++) {
		int i = perm[ii];
		// Build a list of used colors in the nodes neighborhood
		for (int j = 0; j <= k; j++) {
			taken[j] = 0;
		}
		for (int j = 0; j < g.n; j++) {
			numConfChecks++;
			if (i != j && g[i][j]) {
				taken[c[j]]++;
			}
		}
		// if the currently assigned color is legal and not 0, leave it
		// otherwise find a new legal color, and if not possible set it to zero.
		if (c[i] == 0 || taken[c[i]] > 0) {
			int color = 0;
			for (int j = 0; j <= k; j++) {
				if (taken[j] == 0) {
					color = j;
					break;
				}
			}
			c[i] = color;
		}
	}
	delete[] perm;
	delete[] taken;
}

void initializeColoringForTabu(Graph& g, int* c, int k)
{
	// A simple greedy algorithm that leaves the assigned color  if possible, gives another legal color or
	// assigns a random color if nothing else is available.
	// First produce a random permutation for the vertex order
	int* perm = new int[g.n];
	for (int i = 0; i < g.n; i++) {
		perm[i] = i;
	}
	for (int i = 0; i < g.n; i++) {
		int p = rand() % g.n;
		int h = perm[i];
		perm[i] = perm[p];
		perm[p] = h;
	}

	int* taken = new int[k + 1];

	// Insure all colors are in the range [1, ... ,k]
	for (int i = 0; i < g.n; i++) {
		if (c[i]<1 || c[i]>k) c[i] = 1;
	}

	// Go through all nodes
	for (int ii = 0; ii < g.n; ii++) {
		int i = perm[ii];
		// Build a list of used colors in the nodes neighborhood
		for (int j = 1; j <= k; j++) {
			taken[j] = 0;
		}
		for (int j = 0; j < g.n; j++) {
			numConfChecks++;
			if (i != j && g[i][j]) {
				taken[c[j]]++;
			}
		}
		// if the currently assigned color is legal, leave it otherwise find a new legal color, and if not possible
		// set it to a random color.
		if (taken[c[i]] > 0) {
			int color = (rand() % k) + 1;
			for (int j = 1; j <= k; j++) {
				if (taken[j] == 0) {
					color = j;
					break;
				}
			}
			c[i] = color;
		}
	}
	delete[] perm;
	delete[] taken;
}
