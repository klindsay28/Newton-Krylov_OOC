#include <cstring>
#include <string>
#include <climits>
#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>
#include <iomanip>
#include <algorithm>
#include <tuple>
#include <set>
#include <unordered_set>
using namespace std;

//Global Variables-----------------------------------------------------------------------
unsigned long long numConfChecks;
int verbose;

//Struct used for holding the graph------------------------------------------------------
struct Graph {
	int n = 0;
	int m = 0;
	int maxDeg = 0;
	vector<vector<int>> AList;
	vector<int> deg;
};

//Struct used in conjunction with sorting by degree--------------------------------------
struct degItem {
	int deg;
	int vertex;
};
struct maxDeg {
	bool operator() (const degItem& lhs, const degItem& rhs) const {
		//Compares two degItems by degree, then vertex label
		numConfChecks += 2;
		if (lhs.deg > rhs.deg) return true;
		numConfChecks += 2;
		if (lhs.deg < rhs.deg) return false;
		//if we are here we know that lhs.deg == rhs.deg. Our choice can be arbitrary
		if (lhs.vertex > rhs.vertex) return true;
		else return false;
	}
};

//Struct used in conjunction with the DSatur priority queue-------------------------------
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

//Procedures used by all algorithms------------------------------------------------------
void logAndExit(string s) {
	//Writes message s to screen and log file and then exits the program
	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << s;
	cout << s;
	resultsLog.close();
	exit(1);
}
Graph readInputFile(char fname[]) {
	//Reads a DIMACS format file and return the corresponding Graph struct
	ifstream inStream;
	inStream.open(fname);
	if (inStream.fail()) logAndExit("Error. Unrecognized argument/filename <" + string(fname) + "> in command.\n");
	char c, str[1000];
	int line = 0, u, v, edgeCnt = 0;
	vector<vector<bool>> A;
	Graph G;
	try {
		while (!inStream.eof()) {
			line++;
			inStream.get(c);
			if (inStream.eof()) break;
			switch (c) {
			case 'c':
				//ignore a comment line
				inStream.putback('c');
				inStream.get(str, 999, '\n');
				break;
			case 'p':
				//read the parameter line of the file and set up the adjacency matrix
				inStream.get(c);
				inStream.getline(str, 999, ' ');
				if (G.n != 0 || G.m != 0) logAndExit("Invalid input file. Line starting with 'p' at line " + to_string(line) + " defined more than once.\n");
				if (strcmp(str, "edge")) logAndExit("Invalid input file. Problem at line " + to_string(line) + ". No 'edge' keyword found.\n");
				inStream >> G.n >> G.m;
				A.clear();
				A.resize(G.n, vector<bool>(G.n, false));
				for (u = 0; u < G.n; u++) A[u][u] = true;
				break;
			case 'e':
				//Read an edge
				inStream >> u >> v;
				if (u < 1 || u > G.n || v < 1 || v > G.n || u == v) logAndExit("Invalid input file. Problem at line " + to_string(line) + ". Invalid edge.\n");
				if (!A[u - 1][v - 1]) edgeCnt++;
				else logAndExit("Invalid input file. Problem at line " + to_string(line) + ". Edge defined previously.\n");
				A[u - 1][v - 1] = true;
				A[v - 1][u - 1] = true;
				break;
			default:
				logAndExit("Invalid input file. Problem at line " + to_string(line) + "\n");
			}
			inStream.get();
		}
		//Finished reading the file
		inStream.close();
		if (edgeCnt != G.m) logAndExit("Invalid input file. Number of read edges does not equal number specified at the top of the file.\n");
	}
	catch (...) {
		logAndExit("Invalid input file. Unidentified error near line " + to_string(line) + ".\n");
	}
	inStream.close();
	//Check to see if there are no edges. If so, exit straight away
	if (G.m <= 0) {
		logAndExit("Graph has no edges. Optimal solution is obviously using one colour.\n");
	}
	//Now use the adjacency matrix to construct the graph G
	G.deg.clear();
	G.AList.clear();
	G.deg.resize(G.n, 0);
	G.AList.resize(G.n, vector<int>());
	for (u = 0; u < G.n; u++) {
		for (v = 0; v < G.n; v++) {
			if (A[u][v] && u != v) {
				G.AList[u].push_back(v);
				G.deg[u]++;
			}
		}
	}
	G.maxDeg = *max_element(G.deg.begin(), G.deg.end());
	return(G);
}
void prettyPrintSolution(vector<vector<int>>& S) {
	int i, count = 0, col;
	cout << "\n\n";
	for (col = 0; col < S.size(); col++) {
		cout << "C-" << col << "\t= {";
		if (S[col].size() == 0) cout << "empty}\n";
		else {
			for (i = 0; i < S[col].size() - 1; i++) cout << S[col][i] << ", ";
			cout << S[col][S[col].size() - 1] << "}\n";
			count = count + S[col].size();
		}
	}
	cout << "Total Number of Nodes = " << count << endl;
}

//Functions for Greedy and DSatur Algorithm----------------------------------------------
int getFirstFeasCol(Graph& G, int v, vector<int>& c, vector<bool>& used) {
	int i;
	for (int u : G.AList[v]) {
		if (c[u] != -1) used[c[u]] = true;
	}
	for (i = 0; i < used.size(); i++) {
		if (used[i] == false) break;
	}
	for (int u : G.AList[v]) {
		if (c[u] != -1) used[c[u]] = false;
	}
	numConfChecks += G.deg[v] + G.deg[v];
	return i;
}
vector<int> greedycol(Graph& G, bool sortByDegree) {
	int i;
	vector<bool> used(G.maxDeg + 1, false);
	vector<int> c(G.n, -1), perm;
	if (sortByDegree) {
		//Sort vertices by degree in O(n lg n) time
		set<degItem, maxDeg> L;
		numConfChecks += G.n;
		for (i = 0; i < G.n; i++) L.insert({ G.deg[i], i });
		for (degItem el : L) perm.push_back(el.vertex);
	}
	else {
		//Shuffle the vertices in O(n) time
		for (i = 0; i < G.n; i++) perm.push_back(i);
		random_shuffle(perm.begin(), perm.end());
	}
	//Do the greedy algorithm using perm
	for (int v : perm) {
		i = getFirstFeasCol(G, v, c, used);
		c[v] = i;
	}
	return c;
}
vector<int> DSatur(Graph& G) {
	int u, i;
	vector<bool> used(G.maxDeg + 1, false);
	vector<int> c(G.n), d(G.n);
	vector<set<int>> adjCols(G.n);
	set<satItem, maxSat> Q;
	set<satItem, maxSat>::iterator maxPtr;
	//Initialise the the data structures. These are a (binary heap) priority queue, a set of colours adjacent to each uncoloured vertex (initially empty)
	//and the degree d(v) of each uncoloured vertex in the graph induced by uncoloured vertices
	numConfChecks += G.n;
	for (u = 0; u < G.n; u++) {
		c[u] = -1;
		d[u] = G.deg[u];
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
		i = getFirstFeasCol(G, u, c, used);
		c[u] = i;
		//Update the saturation degrees and d-value of all uncoloured neighbours; hence modify their corresponding elements in the priority queue
		numConfChecks += G.deg[u];
		for (int v : G.AList[u]) {
			if (c[v] == -1) {
				Q.erase({ int(adjCols[v].size()), d[v], v });
				adjCols[v].insert(i);
				d[v]--;
				Q.emplace(satItem{ int(adjCols[v].size()), d[v], v });
			}
		}
	}
	return c;
}

//Functions for Greedy IS algorithm------------------------------------------------------
Graph relabelVertices(Graph& G, vector<int>& P) {
	//Create a copy of G with vertices relabelled according to P
	Graph H;
	H.n = G.n, H.m = G.m, H.maxDeg = G.maxDeg;
	H.AList.resize(G.n, vector<int>());
	H.deg.resize(G.n, 0);
	for (int u = 0; u < G.n; u++) {
		numConfChecks += G.deg[u];
		for (int v : G.AList[u]) {
			H.AList[P[u]].push_back(P[v]);
			H.deg[P[u]]++;
		}
	}
	return H;
}
void updateSets(Graph& G, unordered_set<int>& X, unordered_set<int>& Y, vector<int>& c, int u) {
	//Remove u from X (it is now coloured) and move all uncoloured neighbours of u from X to Y
	X.erase(u);
	numConfChecks += G.deg[u];
	for (int v : G.AList[u]) {
		if (c[v] == -1) {
			X.erase(v);
			Y.insert(v);
		}
	}
}
vector<int> greedyIS(Graph& G) {
	//Create a graph H. This is a copy of G with vertices randomly relabelled.
	vector<int> P(G.n);
	for (int v = 0; v < G.n; v++) P[v] = v;
	random_shuffle(P.begin(), P.end());
	Graph H = relabelVertices(G, P);
	//Now colour H
	vector<int> c(H.n, -1);
	unordered_set<int> X, Y;
	for (int v = 0; v < H.n; v++) X.insert(v);
	int i = 0, u;
	while (!X.empty()) {
		//Constructing colour class i.
		while (!X.empty()) {
			u = *X.begin();
			c[u] = i;
			updateSets(H, X, Y, c, u);
		}
		X.swap(Y);
		i++;
	}
	//Convert the vertex labels used in H to correspond to a solution for G.
	vector<int> final(G.n);
	for (int v = 0; v < G.n; v++) final[v] = c[P[v]];
	return final;
}

//Functions for RLF Algorithm------------------------------------------------------------
void populateNeighbourArrays(Graph& G, unordered_set<int>& X, vector<int>& NInX, vector<int>& NInY) {
	for (int u : X) {
		NInX[u] = 0;
		NInY[u] = 0;
	}
	for (int u : X) {
		numConfChecks += G.deg[u];
		for (int v : G.AList[u]) {
			if (X.count(v) == 1) NInX[u]++;
		}
	}
}
int chooseFirstVertex(unordered_set<int>& X, vector<int>& NInX) {
	//Select the vertex in (non-empty) X that has the maximum number of neighbours in X
	int v = -1, max = -1;
	for (int u : X) {
		if (NInX[u] > max) {
			max = NInX[u];
			v = u;
		}
	}
	return v;
}
int chooseNextVertex(unordered_set<int>& X, vector<int>& NInY, vector<int>& NInX) {
	//Select vertex in (non-empty) X with max neighbours in Y; break ties according to minimum neighbours within X
	int v = -1, max = -1, min = INT_MAX;
	for (int u : X) {
		if ((NInY[u] > max) || (NInY[u] == max && NInX[u] < min)) {
			max = NInY[u];
			min = NInX[u];
			v = u;
		}
	}
	return(v);
}
void updateSetsAndDegrees(Graph& G, unordered_set<int>& X, unordered_set<int>& Y, vector<int>& NInX, vector<int>& NInY, unordered_set<int>& D2, vector<int>& c, int u) {
	//Remove u from X (it is now coloured)
	X.erase(u);
	//Move all uncoloured neighbours of u from X to Y
	numConfChecks += G.deg[u];
	for (int v : G.AList[u]) {
		if (c[v] == -1) {
			X.erase(v);
			Y.insert(v);
		}
	}
	//The remaining parts of this procedure now recalculate the contets of NinX and NinY. First calculate a set D2 of all uncoloured vertites within distance two of u.
	D2.clear();
	numConfChecks += G.deg[u];
	for (int v : G.AList[u]) {
		if (c[v] == -1) {
			D2.insert(v);
			numConfChecks += G.deg[v];
			for (int w : G.AList[v]) {
				if (c[w] == -1) {
					D2.insert(w);
				}
			}
		}
	}
	//For each vertex v in D2, now recalculate the number of (uncoloured) neighbours in X and Y
	for (int v : D2) {
		NInX[v] = 0;
		NInY[v] = 0;
		numConfChecks += G.deg[v];
		for (int w : G.AList[v]) {
			if (c[w] == -1) {
				if (X.count(w) == 1) NInX[v]++;
				else if (Y.count(w) == 1) NInY[v]++;
			}
		}
	}
}
vector<int> RLF(Graph& G) {
	vector<int> NInX(G.n), NInY(G.n), c(G.n, -1);
	unordered_set<int> X, Y, D2;
	for (int v = 0; v < G.n; v++) X.insert(v);
	int u, i = 0;
	while (!X.empty()) {
		//Constructing colour class i. First calculate the contents of the neighbours arrays, then colour the vertex u in X that has the most neighbours in X
		populateNeighbourArrays(G, X, NInX, NInY);
		u = chooseFirstVertex(X, NInX);
		c[u] = i;
		updateSetsAndDegrees(G, X, Y, NInX, NInY, D2, c, u);
		while (!X.empty()) {
			//Colour the vertex u in X that has the largest number of neighbours in Y. Break ties according to the minimum neighbours within X
			u = chooseNextVertex(X, NInY, NInX);
			c[u] = i;
			updateSetsAndDegrees(G, X, Y, NInX, NInY, D2, c, u);
		}
		//Have finished constructing colour i
		X.swap(Y);
		i++;
	}
	return c;
}

//Main algorithm-------------------------------------------------------------------------
int main(int argc, char** argv) {

	if (argc <= 1) {
		cout << "Constructive Algorithms for Graph Colouring\n\n"
			<< "USAGE:\n"
			<< "<InputFile>     (Required. File must be in DIMACS format)\n"
			<< "-a <int>        (Algorithm choice: 1 = Greedy (random vertex permutation) (default)\n"
			<< "                                   2 = Greedy (descending vertex degrees / Welsh-Powell algorithm)\n"
			<< "                                   3 = DSatur\n"
			<< "                                   4 = Greedy-IS (random vertex permutation)\n"
			<< "                                   5 = RLF)\n"
			<< "-r <int>        (Random seed. DEFAULT = 1)\n"
			<< "-v              (Verbosity. If present, output is sent to screen. If -v is repeated, more output is given.)\n"
			<< "****\n";
		exit(1);
	}

	int i, randomSeed = 1, algChoice = 1;
	numConfChecks = 0;
	verbose = 0;
	Graph G;
	try {
		for (i = 1; i < argc; i++) {
			if (strcmp("-r", argv[i]) == 0) {
				randomSeed = atoi(argv[++i]);
			}
			else if (strcmp("-a", argv[i]) == 0) {
				algChoice = atoi(argv[++i]);
			}
			else if (strcmp("-v", argv[i]) == 0) {
				verbose++;
			}
			else {
				//Set up input file, read, and close (input must be in DIMACS format)
				G = readInputFile(argv[i]);
			}
		}
	}
	catch (...) {
		logAndExit("Invalid parameters. Check usage and try again.\n");
	}

	//Set Random Seed and start the timer
	srand(randomSeed);
	clock_t runStart = clock();

	//Carry out the chosen algorithm
	vector<int> c;
	if (algChoice == 1)			c = greedycol(G, false);
	else if (algChoice == 2)	c = greedycol(G, true);
	else if (algChoice == 3)	c = DSatur(G);
	else if (algChoice == 4)	c = greedyIS(G);
	else						c = RLF(G);

	//Stop the timer.
	int duration = (int)(((clock() - runStart) / double(CLOCKS_PER_SEC)) * 1000);

	//Convert the solution to the partition representation for display
	int k = *max_element(c.begin(), c.end()) + 1;
	vector<vector<int>> S(k, vector<int>());
	for (i = 0; i < G.n; i++) S[c[i]].push_back(i);
	if (verbose >= 1) cout << " COLS     CPU-TIME(ms)\tCHECKS" << endl;
	if (verbose >= 1) cout << setw(5) << k << setw(11) << duration << "ms\t" << numConfChecks << endl;
	if (verbose >= 2) prettyPrintSolution(S);

	//output the solution to a text file
	ofstream solStrm;
	solStrm.open("solution.txt");
	solStrm << G.n << "\n";
	for (i = 0; i < G.n; i++) solStrm << c[i] << "\n";
	solStrm.close();

	if (algChoice == 1)			logAndExit("Greedy\t" + to_string(k) + "\t" + to_string(duration) + "\t" + to_string(numConfChecks) + "\n");
	else if (algChoice == 2)	logAndExit("Welsh-Powell\t" + to_string(k) + "\t" + to_string(duration) + "\t" + to_string(numConfChecks) + "\n");
	else if (algChoice == 3)	logAndExit("DSatur\t" + to_string(k) + "\t" + to_string(duration) + "\t" + to_string(numConfChecks) + "\n");
	else if (algChoice == 4)	logAndExit("GreedyIS\t" + to_string(k) + "\t" + to_string(duration) + "\t" + to_string(numConfChecks) + "\n");
	else						logAndExit("RLF\t" + to_string(k) + "\t" + to_string(duration) + "\t" + to_string(numConfChecks) + "\n");

	return(0);
}
