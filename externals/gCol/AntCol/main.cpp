/******************************************************************************/
//  This code implements the Ant Colony Optimisation-based algorithm for
//  graph colouring described by Thompson and Dowsland.
//  The local search routines are based on the tabu search algorithm written
//  by Ivo Bloechliger http://rose.epfl.ch/~bloechli/coloring/
//  The remaining code was written by R. Lewis www.rhydLewis.eu
//
//	See: Lewis, R. (2015) A Guide to Graph Colouring: Algorithms and Applications. Berlin, Springer.
//       ISBN: 978-3-319-25728-0. http://www.springer.com/us/book/9783319257280
//
//	for further details
/******************************************************************************/

#include "Graph.h"
#include "inputGraph.h"
#include "tabu.h"
#include "makesolution.h"
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <iomanip>
#include <time.h>
#include <limits.h>

//This makes sure the compiler uses _strtoui64(x, y, z) with Microsoft Compilers, otherwise strtoull(x, y, z) is used
#ifdef _MSC_VER
#define strtoull(x, y, z) _strtoui64(x, y, z)
#endif

using namespace std;

//GLOBAL VARIABLES
unsigned long long numConfChecks;

void logAndExitNow(string s) {
	//Writes message s to screen and log file and then exits the program
	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << s;
	cout << s;
	resultsLog.close();
	exit(1);
}

void usage() {
	cout << "AntCol Algorithm for Graph Colouring\n\n"
		<< "USAGE:\n"
		<< "<InputFile>     (Required. File must be in DIMACS format)\n"
		<< "-s <int>        (Stopping criteria expressed as number of constraint checks. Can be anything up to 9x10^18. DEFAULT = 100,000,000.)\n"
		<< "-I <int>        (Number of iterations of tabu search per cycle. This figure is multiplied by the graph size |V|. DEFAULT = 2)\n"
		<< "-r <int>        (Random seed. DEFAULT = 1)\n"
		<< "-T <int>        (Target number of colours. Algorithm halts if this is reached. DEFAULT = 1.)\n"
		<< "-v              (Verbosity. If present, output is sent to screen. If -v is repeated, more output is given.)\n"
		<< "****\n";
	exit(1);
}

//MAIN PROCEDURE---------------------------------------------------
int main(int argc, char** argv) {

	if (argc <= 1) {
		usage();
	}

	//Read in any input paramters from the command line
	Graph g;
	int i, j, verbose = 0, duration;
	int maxIterations = 2;
	int randomSeed = 1;
	unsigned long long maxChecks = 100000000;
	int targetCols = 2;
	try {
		for (i = 1; i < argc; i++) {
			if (strcmp("-I", argv[i]) == 0) {
				maxIterations = atoi(argv[++i]);
			}
			else if (strcmp("-r", argv[i]) == 0) {
				randomSeed = atoi(argv[++i]);
			}
			else if (strcmp("-s", argv[i]) == 0) {
				maxChecks = strtoull(argv[++i], NULL, 10);
			}
			else if (strcmp("-T", argv[i]) == 0) {
				targetCols = atoi(argv[++i]);
			}
			else if (strcmp("-v", argv[i]) == 0) {
				verbose++;
			}
			else {
				inputDimacsGraph(g, argv[i]);
				cout << "Ant Colony Optimisation Algorithm for Graph Colouring using <" << argv[i] << ">\n\n";
			}
		}
	}
	catch (...) {
		logAndExitNow("Error with input parameters. Exiting.\n");
	}

	if (targetCols < 2 || targetCols > g.n) targetCols = 2;

	//Now set up some output files, and output some details to the screen
	ofstream colTimeStream, colConfStream;
	colTimeStream.open("teffort.txt"); colConfStream.open("ceffort.txt");
	if (colTimeStream.fail() || colConfStream.fail()) {
		logAndExitNow("ERROR OPENING output files");
	}

	//Do a check to see if we have the empty graph. If so, end immediately.
	if (g.nbEdges <= 0) {
		colConfStream << "1\t0\n0\tX\t0\n";
		colTimeStream << "1\t0\n0\tX\t0\n";
		colConfStream.close();
		colTimeStream.close();
		logAndExitNow("Graph has no edges. Optimal solution is obviously using one colour. Exiting.\n");
	}

	//Initialise list of neighbors for each node (two structures used: archaic notation with col 0 representing cardinality (for tabu)
	//and using vectors in normal notation. Also make the degree vector
	int** neighbors = new int* [g.n];
	vector<vector<int> > neighbours(g.n, vector<int>());
	vector<int> degree(g.n);
	for (i = 0; i < g.n; i++) {
		neighbors[i] = new int[g.n + 1];
		neighbors[i][0] = 0;
	}
	for (i = 0; i < g.n; i++) {
		for (j = 0; j < g.n; j++) {
			if (g[i][j] && i != j) {
				neighbors[i][++neighbors[i][0]] = j;
				neighbours[i].push_back(j);
			}
		}
	}
	for (i = 0; i < g.n; i++) degree[i] = neighbours[i].size();

	//Initialise some further parameters
	numConfChecks = 0;
	maxIterations = g.n * maxIterations;
	if (maxChecks < 0) maxChecks = 1000000;
	clock_t clockStart = clock();
	srand(randomSeed);
	double alpha = 2.0, beta = 3.0, evap = 0.75, solCost;
	int nants = 10, multisets = 5, k = g.n, clashes = 0, *solByNode = new int[g.n], ant;
	bool SIsFeasible;
	vector< vector<int> > bestSoFar, S;
	for (i = 0; i < g.n; i++) bestSoFar.push_back({ i });

	//Output some details to the screen
	if (verbose >= 1) cout << " COLS     CPU-TIME\tCHECKS" << endl;

	//Initialise the pheremone matrix t and declare delta
	vector< vector<double> > t(g.n, vector<double>(g.n, 1.0));
	vector< vector<double> > delta(g.n, vector<double>(g.n));

	//Declare some other vectors used in the build procedure
	vector<int> X, Y;
	vector< vector<int> > tempX(multisets, vector<int>()), tempY(multisets, vector<int>()), ISet(multisets, vector<int>());
	vector<double> tauEta;

	//MAIN LOOP OF ANTS ALGORITHM------------------------------------------------------------------
	while (numConfChecks < maxChecks) {

		//Initialise the local pheremone matrix delta
		for (i = 0; i < g.n; i++) {
			for (j = 0; j < g.n; j++) {
				delta[i][j] = 0.0;
			}
		}

		//nants cycle. Seeking solutions using k colours
		for (ant = 0; ant < nants; ant++) {

			//Declare a new empty solution S, and construct it using a maximum of k colours.
			S.clear();
			S.resize(k, vector<int>());
			SIsFeasible = buildSolution(g, S, neighbours, degree, t, k, alpha, beta, multisets, X, Y, tempX, tempY, ISet, tauEta);

			if (!SIsFeasible) {
				//Solution using |S| = k colours not feasible. Run tabu search
				for (i = 0; i < S.size(); i++) for (j = 0; j < S[i].size(); j++) solByNode[S[i][j]] = i + 1;
				clashes = tabu(g, neighbors, solByNode, k, maxIterations, 0);
				if (clashes == 0) SIsFeasible = true;
				if (verbose >= 2) cout << "          -> Solution" << setw(5) << ant << " constructed. k = " << k << " Cost = " << clashes << endl;
				for (i = 0; i < S.size(); i++) S[i].clear();
				for (i = 0; i < g.n; i++) S[solByNode[i] - 1].push_back(i);
			}
			else {
				if (verbose >= 2) cout << "          -> Solution" << setw(5) << ant << " constructed. k = " << k << " Cost = 0" << endl;
			}

			//Calculate the cost of the solution and update delta
			if (SIsFeasible)	solCost = 3.0;
			else				solCost = 1.0 / double(clashes);
			for (i = 0; i < S.size(); i++) {
				for (j = 0; j < S[i].size() - 1; j++) {
					for (int j1 = j + 1; j1 < S[i].size(); j1++) {
						delta[S[i][j]][S[i][j1]] += solCost;
						delta[S[i][j1]][S[i][j]] += solCost;
					}
				}
			}

			if (SIsFeasible) {
				// A feasible solution using |S| colours (where |S| <= k) has been found.
				if (S.size() < bestSoFar.size()) {
					//This is the best solution found so far. Output details to the log files and end if the target has been reached
					bestSoFar = S;
					duration = (int)(((clock() - clockStart) / double(CLOCKS_PER_SEC)) * 1000);
					if (verbose >= 1) cout << setw(5) << bestSoFar.size() << setw(11) << duration << "ms\t" << numConfChecks << endl;
					colConfStream << bestSoFar.size() << "\t" << numConfChecks << "\n";
					colTimeStream << bestSoFar.size() << "\t" << duration << "\n";
				}
				//Because we have found a feasible solution S using k (or fewer) colours we can leave this nants cycle and set k = |S| - 1
				break;
			}

			//Also leave the nants cycle if we have exceeded the checks limit
			if (numConfChecks >= maxChecks) break;

		} //End of nants cycle

		if (bestSoFar.size() <= targetCols) {
			//Have reached the target, so end
			break;
		}
		else {
			//Otherwise, update the t matrix and continue
			for (i = 0; i < g.n; i++) {
				for (j = 0; j < g.n; j++) {
					if (i != j) {
						t[i][j] = evap * t[i][j] + delta[i][j];
					}
				}
			}
			k = bestSoFar.size() - 1;
		}
	}
	// END OF MAIN LOOP OF ANTS ALGORITHM------------------------------------------------------------------

	//Write final details to output stream
	duration = int(((double)(clock() - clockStart) / CLOCKS_PER_SEC) * 1000);
	if (numConfChecks >= maxChecks) {
		colConfStream << k << "\t" << "X\t" << numConfChecks << "\n";
		colTimeStream << k << "\t" << "X\t" << duration << "\n";
		if (verbose >= 1) cout << "\nRun limit exceeded. No solution using " << k << " colours was achieved (Checks = " << numConfChecks << ", " << duration << "ms)" << endl;
	}
	else {
		//Target was reached
		colConfStream << "1\t" << "X" << "\n";
		colTimeStream << "1\t" << "X" << "\n";
		if (verbose >= 1) cout << "\nSolution with <=" << targetCols << " colours has been found. Ending..." << endl;
	}
	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << "AntCol\t" << bestSoFar.size() << "\t" << duration << "\t" << numConfChecks << endl;
	resultsLog.close();

	//output the solution to a text file
	ofstream solStrm;
	solStrm.open("solution.txt");
	vector<int> grp(g.n);
	for (i = 0; i < bestSoFar.size(); i++) { for (int j = 0; j < bestSoFar[i].size(); j++) { grp[bestSoFar[i][j]] = i; } }
	solStrm << g.n << "\n";
	for (i = 0; i < g.n; i++) solStrm << grp[i] << "\n";
	solStrm.close();

	colConfStream.close();
	colTimeStream.close();

	//Delete Arrays
	for (i = 0; i < g.n; i++) delete[] neighbors[i];
	delete[] neighbors;
	delete[] solByNode;

	return(0);
}
