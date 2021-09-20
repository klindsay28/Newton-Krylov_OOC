#include "makesolution.h"

int ASSIGNED = INT_MIN;

extern unsigned long long numConfChecks;

//-------------------------------------------------------------------------------------
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
int generateInitialK(Graph& g, int alg, vector<int>& bestColouring) {
	//Produce an solution using a constructive algorithm to get an intial setting for k
	int i, j;

	//Make the structures needed for the constructive algorithms
	vector< vector<int> > candSol, adjList(g.n, vector<int>());
	vector<int> colNode(g.n, INT_MAX);
	for (i = 0; i < g.n; i++) {
		unsigned char* uc_ptr = g[i];
		for (j = 0; j < g.n; j++) {
			if (uc_ptr[j] && i != j) {
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


//-----------BELOW ARE THE FUNCTIONS FOR GENERATING SOLUTIONS WITH A MAXIMUM K COLOURS
inline
void updateColOptions(vector< vector<bool> >& availCols, vector<int>& numColOptions, Graph& g, int v, int col)
{
	int i;
	//Updates colOptions vector due to node v being assigned a colour
	numColOptions[v] = ASSIGNED;
	for (i = 0; i < g.n; i++) {
		unsigned char* uc_ptr = g[i];
		if (numColOptions[i] != ASSIGNED) {
			numConfChecks++;
			if (availCols[i][col] && uc_ptr[v]) {
				availCols[i][col] = false;
				numColOptions[i]--;
			}
		}
	}
}
inline
bool coloursAvailable(vector<int>& colOptions)
{
	int i;
	for (i = 0; i < colOptions.size(); i++) {
		if (colOptions[i] >= 1)
			return true;
	}
	return false;
}
inline
int chooseNextNode(vector<int>& colOptions)
{
	int i;
	int minOptions = INT_MAX;
	vector<int> a;
	for (i = 0; i < colOptions.size(); i++) {
		if (colOptions[i] != ASSIGNED) {
			if (colOptions[i] >= 1) {
				if (colOptions[i] < minOptions) {
					a.clear();
					a.push_back(i);
					minOptions = colOptions[i];
				}
				else if (colOptions[i] == minOptions) {
					a.push_back(i);
				}
			}
		}
	}
	int x = rand() % a.size();
	return(a[x]);
}
inline
int assignToColour(vector< vector<bool> >& availCols, vector< vector<int> >& candSol, int k, int v)
{
	int c = 0;
	while (c < k) {
		if (availCols[v][c]) {
			//colour c is OK for vertex v, so we assign it and exit
			candSol[c].push_back(v);
			return c;
		}
		c++;
	}
	return(-1);
}
void makeInitSolution(Graph& g, vector<int>& sol, int k, int verbose)
{
	int i, v, j, c;

	//1) Make a 2D vector containing all colour options for each node (initially k for all nodes)
	vector<int> numColOptions(g.n, k);
	vector< vector<bool> > availCols(g.n, vector<bool>(k, true));

	//... and make an empty solution in convienient representation
	vector< vector<int> > candSol(k, vector<int>());

	//2) Now add a random node to the first colour and update colOptions
	v = rand() % g.n;
	c = assignToColour(availCols, candSol, k, v);
	updateColOptions(availCols, numColOptions, g, v, c);
	//3) For each remaining node with available colour options, choose a node with minimal (>=1) options and assign to an early colour
	while (coloursAvailable(numColOptions)) {
		//choose node to colour
		v = chooseNextNode(numColOptions);
		//assign to a colour
		c = assignToColour(availCols, candSol, k, v);
		updateColOptions(availCols, numColOptions, g, v, c);
	}

	//When we are here, we either have a full valid solution, or some nodes are still unplaced (marked with 0's in numColOptions)
	//These are now placed in random colours
	for (i = 0; i < g.n; i++) {
		if (numColOptions[i] == 0) {
			//put node i into a random colour
			candSol[rand() % k].push_back(i);
		}
	}

	//3) Now tranfer to the more convienient representation in the population itself and end
	for (i = 0; i < k; i++) {
		for (j = 0; j < candSol[i].size(); j++) {
			sol[candSol[i][j]] = i + 1;
		}
	}
}

//-------------------------------------------------------------------------------------
