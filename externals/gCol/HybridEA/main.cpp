/******************************************************************************/
//  This code implements the Hybrid Evolutionary Algorithm of Galinier and Hao.
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
#include "xover.h"
#include "diversity.h"
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <vector>
#include <iomanip>

//This makes sure the compiler uses _strtoui64(x, y, z) with Microsoft Compilers, otherwise strtoull(x, y, z) is used
#ifdef _MSC_VER
#define strtoull(x, y, z) _strtoui64(x, y, z)
#endif

using namespace std;

void logAndExitNow(string s) {
	//Writes message s to screen and log file and then exits the program
	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << s;
	cout << s;
	resultsLog.close();
	exit(1);
}

int ASS = INT_MIN;
bool solIsOptimal(vector<int>& sol, Graph& g, int k);
void makeAdjList(int** neighbors, Graph& g);
void replace(vector<vector<int> >& population, vector<int>& parents, vector<int>& osp, vector<int>& popCosts, Graph& g, int oCost);

unsigned long long numConfChecks;

void usage() {
	cout << "Hybrid EA for Graph Colouring\n\n"
		<< "USAGE:\n"
		<< "<InputFile>     (Required. File must be in DIMACS format)\n"
		<< "-s <int>        (Stopping criteria expressed as number of constraint checks. Can be anything up to 9x10^18. DEFAULT = 100,000,000.)\n"
		<< "-I <int>        (Number of iterations of tabu search per cycle. This figure is multiplied by the graph size |V|. DEFAULT = 16)\n"
		<< "-r <int>        (Random seed. DEFAULT = 1)\n"
		<< "-T <int>        (Target number of colours. Algorithm halts if this is reached. DEFAULT = 1.)\n"
		<< "-v              (Verbosity. If present, output is sent to screen. If -v is repeated, more output is given.)\n"
		<< "-p <int>        (Population Size. Should be 2 or more. DEFAULT = 10)\n"
		<< "-a <int>        (Choice of construction algorithm to determine initial value for k. DSsatur = 1, Greedy = 2. DEFAULT = 1.)\n"
		<< "-x <int>        (Crossover Operator. 1 = GPX (2 parents)\n"
		<< "                                     2 = GPX (2 parents + Kempe chain mutation)\n"
		<< "                                     3 = MPX (4 parent crossover with q=2)\n"
		<< "                                     4 = GGA (2 parents)\n"
		<< "                                     5 = nPoint (2 parents)\n"
		<< "                 DEFAULT = 1)\n"
		<< "-d              (If present population diversity is measured after each crossover)\n"
		<< "****\n";
	exit(1);
}

int main(int argc, char** argv) {

	if (argc <= 1) {
		usage();
	}

	Graph g;
	int i, k, popSize = 10, maxIterations = 16, verbose = 0, randomSeed = 1, constructiveAlg = 1, targetCols = 2, xOverType = 1;
	bool solFound = false, doKempeMutation = false, measuringDiversity = false;
	unsigned long long maxChecks = 100000000;
	vector<int> parents;

	//This variable keeps count of the number of times information about the instance is looked up
	numConfChecks = 0;

	try {
		for (i = 1; i < argc; i++) {
			if (strcmp("-p", argv[i]) == 0) {
				popSize = atoi(argv[++i]);
				if (popSize < 2) logAndExitNow("Error: PopSize should be >= 2\n");
			}
			else if (strcmp("-a", argv[i]) == 0) {
				constructiveAlg = atoi(argv[++i]);
			}
			else if (strcmp("-I", argv[i]) == 0) {
				maxIterations = atoi(argv[++i]);
			}
			else if (strcmp("-r", argv[i]) == 0) {
				randomSeed = atoi(argv[++i]);
			}
			else if (strcmp("-T", argv[i]) == 0) {
				targetCols = atoi(argv[++i]);
			}
			else if (strcmp("-v", argv[i]) == 0) {
				verbose++;
			}
			else if (strcmp("-s", argv[i]) == 0) {
				maxChecks = strtoull(argv[++i], NULL, 10);
			}
			else if (strcmp("-x", argv[i]) == 0) {
				xOverType = atoi(argv[++i]);
			}
			else if (strcmp("-d", argv[i]) == 0) {
				measuringDiversity = true;
			}
			else {
				cout << "Hybrid Evolutionary Algorithm using <" << argv[i] << ">\n\n";
				inputDimacsGraph(g, argv[i]);
			}
		}
	}
	catch (...) {
		logAndExitNow("Error in input parameters. Exiting...\n");
	}

	//Set the number of parents in each crossover and decide if the Kempe mutation is going to be used
	if (xOverType == 3) parents.resize(4);
	else parents.resize(2);
	if (xOverType == 2) doKempeMutation = true;

	//set tabucol limit
	maxIterations = maxIterations * g.n;
	if (targetCols < 2 || targetCols > g.n) targetCols = 2;

	//Now set up some output files
	ofstream timeStream, confStream;
	timeStream.open("teffort.txt"); confStream.open("ceffort.txt");
	if (timeStream.fail() || confStream.fail()) logAndExitNow("ERROR OPENING output FILE");

	//Do a check to see if we have the empty graph. If so, end immediately.
	if (g.nbEdges <= 0) {
		confStream << "1\t0\n0\tX\t0\n";
		timeStream << "1\t0\n0\tX\t0\n";
		confStream.close();
		timeStream.close();
		logAndExitNow("Graph has no edges. Optimal solution is obviously using one colour. Exiting.");
	}

	//Make the adjacency list structure
	int** neighbors = new int* [g.n];
	makeAdjList(neighbors, g);

	//Produce some output
	if (verbose >= 1) cout << " COLS     CPU-TIME\tCHECKS" << endl;

	//Seed and start timer
	clock_t clockStart = clock();
	srand(randomSeed);
	numConfChecks = 0;

	//Data structures used for population and offspring
	vector<vector<int> > population(popSize, vector<int>(g.n));
	vector<int> popCosts(popSize);
	vector<int> osp(g.n), bestColouring(g.n);

	//Generate the initial value for k using greedy or dsatur algorithm
	k = generateInitialK(g, constructiveAlg, bestColouring);
	//..and write the results to the output file
	int duration = int(((double)(clock() - clockStart) / CLOCKS_PER_SEC) * 1000);
	if (verbose >= 1) cout << setw(5) << k << setw(11) << duration << "ms\t" << numConfChecks << " (via constructive)" << endl;
	confStream << k << "\t" << numConfChecks << "\n";
	timeStream << k << "\t" << duration << "\n";
	if (k <= targetCols) {
		if (verbose >= 1) cout << "\nSolution with  <=" << targetCols << " colours has been found. Ending..." << endl;
		confStream << "1\t" << "X" << "\n";
		timeStream << "1\t" << "X" << "\n";
		ofstream resultsLog("resultsLog.log", ios::app);
		resultsLog << "HEA\t" << targetCols << "\t" << int(((double)(clock() - clockStart) / CLOCKS_PER_SEC) * 1000) << "\t" << numConfChecks << endl;
		resultsLog.close();
	}

	//MAIN ALGORITHM
	k--;
	while (numConfChecks < maxChecks && k + 1 > targetCols) {
		solFound = false;

		//First build the population
		for (i = 0; i < popSize; i++) {
			//Build a solution using modified DSatur algorithm
			makeInitSolution(g, population[i], k, verbose);
			//Check to see whether this solution is alrerady optimal or if the cutoff point has been reached. If so, we end
			if (solIsOptimal(population[i], g, k)) {
				solFound = true;
				for (int j = 0; j < g.n; j++)osp[j] = population[i][j];
				break;
			}
			if (numConfChecks >= maxChecks) {
				for (int j = 0; j < g.n; j++)osp[j] = population[i][j];
				break;
			}
			//Improve each solution via tabu search and record their costs
			popCosts[i] = tabu(g, population[i], k, maxIterations, 0, neighbors);
			//Check to see whether this solution is now optimal or if the cuttoff point is reached. If so, we end
			if (verbose >= 2)cout << "          -> Individual " << setw(4) << i << " constructed. Cost = " << popCosts[i] << endl;
			if (popCosts[i] == 0) {
				solFound = true;
				for (int j = 0; j < g.n; j++)osp[j] = population[i][j];
				break;
			}
			if (numConfChecks >= maxChecks) {
				for (int j = 0; j < g.n; j++)osp[j] = population[i][j];
				break;
			}
		}

		//Now evolve the population
		int rIts = 0, oCost = 1, best = INT_MAX;
		while (numConfChecks < maxChecks && !solFound) {

			//Choose parents and perform crossover to produce a new offspring
			doCrossover(xOverType, osp, parents, g, k, population);

			//Improve the offspring via tabu search and record its cost
			oCost = tabu(g, osp, k, maxIterations, 0, neighbors);

			//Write osp over weaker parent and update popCosts
			replace(population, parents, osp, popCosts, g, oCost);

			if (verbose >= 2) {
				cout << "          -> Offspring " << setw(5) << rIts << " constructed. Cost = " << oCost;
				if (measuringDiversity) cout << "\tDiversity = " << measureDiversity(population, k);
				cout << endl;
			}

			rIts++;

			if (oCost < best) best = oCost;
			if (oCost == 0) solFound = true;
		}

		//Algorithm has finished at this k
		duration = int(((double)(clock() - clockStart) / CLOCKS_PER_SEC) * 1000);
		if (solFound) {
			if (verbose >= 1) cout << setw(5) << k << setw(11) << duration << "ms\t" << numConfChecks << endl;
			confStream << k << "\t" << numConfChecks << "\n";
			timeStream << k << "\t" << duration << "\n";
			//Copy the current solution as the best solution
			for (int i = 0; i < g.n; i++) bestColouring[i] = osp[i] - 1;
			if (k <= targetCols) {
				if (verbose >= 1) cout << "\nSolution with  <=" << targetCols << " colours has been found. Ending..." << endl;
				confStream << "1\t" << "X" << "\n";
				timeStream << "1\t" << "X" << "\n";
				ofstream resultsLog("resultsLog.log", ios::app);
				resultsLog << "HEA\t" << targetCols << "\t" << int(((double)(clock() - clockStart) / CLOCKS_PER_SEC) * 1000) << "\t" << numConfChecks << endl;
				resultsLog.close();
				break;
			}
		}
		else {
			if (verbose >= 1) cout << "\nRun limit exceeded. No solution using " << k << " colours was achieved (Checks = " << numConfChecks << ", " << duration << "ms)" << endl;
			confStream << k << "\tX\t" << numConfChecks << "\n";
			timeStream << k << "\tX\t" << duration << "\n";
			ofstream resultsLog("resultsLog.log", ios::app);
			resultsLog << "HEA\t" << k + 1 << "\t" << int(((double)(clock() - clockStart) / CLOCKS_PER_SEC) * 1000) << "\t" << numConfChecks << endl;
			resultsLog.close();
		}

		k--;
	}

	ofstream solStrm;
	solStrm.open("solution.txt");
	solStrm << g.n << "\n";
	for (int i = 0; i < g.n; i++)solStrm << bestColouring[i] << "\n";
	solStrm.close();
	return(0);
}

//*********************************************************************
inline
bool solIsOptimal(vector<int>& sol, Graph& g, int k)
{
	int i, j;
	for (i = 0; i < (g.n) - 1; i++) {
		for (j = i + 1; j < g.n; j++) {
			if (sol[i] == sol[j] && g[i][j])
				return(false);
		}
	}
	//If we are here then we have established a solution with k or fewer colours
	return(true);
}
//*********************************************************************
void makeAdjList(int** neighbors, Graph& g)
{
	//Makes the adjacency list corresponding to G
	for (int i = 0; i < g.n; i++) {
		neighbors[i] = new int[g.n + 1];
		neighbors[i][0] = 0;
	}
	for (int i = 0; i < g.n; i++) {
		for (int j = 0; j < g.n; j++) {
			if (g[i][j] && i != j) {
				neighbors[i][++neighbors[i][0]] = j;
			}
		}
	}
}

//*********************************************************************
void replace(vector<vector<int> >& population, vector<int>& parents, vector<int>& osp, vector<int>& popCosts, Graph& g, int oCost)
{
	//Go through the parents and ID the worst one
	int toDie = -1, i, max = INT_MIN;
	for (i = 0; i < parents.size(); i++) {
		if (popCosts[parents[i]] > max) {
			max = popCosts[parents[i]];
			toDie = parents[i];
		}
	}
	//Copy osp over the parent selected toDie
	population[toDie] = osp;
	popCosts[toDie] = oCost;
}
