#include "initialsolutiongenerator.h"
#include <limits.h>
#include <set>
#include <algorithm>

extern unsigned long long numConfChecks;
extern vector< vector<bool> > adjacent;
extern vector< vector<int> > adjList;
extern vector<int> degree;

struct satItem {
	int sat;
	int deg;
	int vertex;
};
struct maxSat {
	bool operator() (const satItem& lhs, const satItem& rhs) const {
		//Compares two satItems sat deg, then degree, then vertex label
		numConfChecks += 2;
		if (lhs.sat > rhs.sat) return true;
		numConfChecks += 2;
		if (lhs.sat < rhs.sat) return false;
		//if we are we know that lhs.sat == rhs.sat
		numConfChecks += 2;
		if (lhs.deg > rhs.deg) return true;
		numConfChecks += 2;
		if (lhs.deg < rhs.deg) return false;
		//if we are here we know that lhs.sat == rhs.sat and lhs.deg == rhs.deg. Our choice can be arbitrary
		if (lhs.vertex > rhs.vertex) return true;
		else return false;
	}
};
int getFirstFeasCol(int v, vector<int>& c, vector<bool>& used) {
	int i;
	for (int u : adjList[v]) {
		if (c[u] != -1) used[c[u]] = true;
	}
	for (i = 0; i < used.size(); i++) {
		if (used[i] == false) break;
	}
	for (int u : adjList[v]) {
		if (c[u] != -1) used[c[u]] = false;
	}
	numConfChecks += degree[v] + degree[v];
	return i;
}
void DSatur(int numNodes, vector<int> &colNode) {
	//This constructs a solution using the DSatur algorithm.
	int u, i;
	vector<bool> used(numNodes, false);
	vector<int> d(numNodes);
	vector<set<int>> adjCols(numNodes);
	set<satItem, maxSat> Q;
	set<satItem, maxSat>::iterator maxPtr;
	//Initialise the the data structures. These are a (binary heap) priority queue, a set of colours adjacent to each uncoloured vertex (initially empty)
	//and the degree d(v) of each uncoloured vertex in the graph induced by uncoloured vertices
	numConfChecks += numNodes;
	for (u = 0; u < numNodes; u++) {
		colNode[u] = -1;
		d[u] = degree[u];
		adjCols[u] = set<int>();
		Q.emplace(satItem{ 0, d[u], u });
	}
	//DSatur algorithm
	while (!Q.empty()) {
		//Get the vertex u with highest saturation degree, breaking ties with d. Remove it from the priority queue and colour it
		numConfChecks++;
		maxPtr = Q.begin();
		u = (*maxPtr).vertex;
		Q.erase(maxPtr);
		i = getFirstFeasCol(u, colNode, used);
		colNode[u] = i;
		//Update the saturation degrees and d-value of all uncoloured neighbours; hence modify their corresponding elements in the priority queue
		numConfChecks += degree[u];
		for (int v : adjList[u]) {
			if (colNode[v] == -1) {
				Q.erase({ int(adjCols[v].size()), d[v], v });
				adjCols[v].insert(i);
				d[v]--;
				Q.emplace(satItem{ int(adjCols[v].size()), d[v], v });
			}
		}
	}
}

//-------------------------------------------------------------------------------------
void makeInitSolution(int numNodes, vector< vector<int> >& candSol, vector< vector<int> >& tempSol, vector<int>& colNode)
{
	//Colour the nodes using the DSatur algorithm
	DSatur(numNodes, colNode);
	int k = *max_element(colNode.begin(), colNode.end()) + 1;
	candSol.resize(k, vector<int>());
	for (int u = 0; u < numNodes; u++) {
		candSol[colNode[u]].push_back(u);
	}
	//Also copy candSol to TempSol
	tempSol = candSol;
}
