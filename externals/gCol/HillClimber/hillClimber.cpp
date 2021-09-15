/******************************************************************************/
//  This code implements the Hill Climbing method for graph colouring.
//  The code was written by R. Lewis www.rhydLewis.eu
//
//	See: Lewis, R. (2015) A Guide to Graph Colouring: Algorithms and Applications. Berlin, Springer.
//       ISBN: 978-3-319-25728-0. http://www.springer.com/us/book/9783319257280
//
//	for further details
/******************************************************************************/


#include "hillClimber.h"
#include<string.h>

//This makes sure the compiler uses _strtoui64(x, y, z) with Microsoft Compilers, otherwise strtoull(x, y, z) is used
#ifdef _MSC_VER
#define strtoull(x, y, z) _strtoui64(x, y, z)
#endif

//GLOBAL VARIABLES THAT KEEP TRACK OF THE NUMBER OF CONSTRAINT CHECKS AND PROBLEM INFO
unsigned long long numConfChecks;
vector< vector<bool> > adjacent;
vector< vector<int> > adjList;
vector<int> degree;

void logAndExitNow(string s) {
	//Writes message s to screen and log file and then exits the program
	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << s;
	cout << s;
	resultsLog.close();
	exit(1);
}

void usage() {
	cout << "Hill Climbing Algorithm for Graph Colouring\n\n"
		<< "USAGE:\n"
		<< "<InputFile>     (Required. File must be in DIMACS format)\n"
		<< "-s <int>        (Stopping criteria expressed as number of constraint checks. Can be anything up to 9x10^18. DEFAULT = 100,000,000.)\n"
		<< "-I <int>        (Number of iterations of local search per cycle. DEFAULT = 1000)\n"
		<< "-r <int>        (Random seed. DEFAULT = 1)\n"
		<< "-T <int>        (Target number of colours. Algorithm halts if this is reached. DEFAULT = 1.)\n"
		<< "-v              (Verbosity. If present, output is sent to screen. If -v is repeated, more output is given.)\n"
		<< "****\n";
	exit(0);
}

int main(int argc, char** argv) {

	if (argc <= 1) {
		usage();
	}

	//PARAMETERS----------------------------------------------------------------------------------
	//VARIABLES (DEFAULT VALS)
	int numNodes = 0, numEdges = 0, i, duration;
	unsigned long long maxConfChecks = 100000000;
	int itLimit = 1000;
	int seed = 1;
	int targetCols = 2;
	int verbose = 0;

	try {
		for (i = 1; i < argc; i++) {
			if (strcmp("-s", argv[i]) == 0) {
				maxConfChecks = strtoull(argv[++i], NULL, 10);
			}
			else if (strcmp("-I", argv[i]) == 0) {
				itLimit = atoi(argv[++i]);
			}
			else if (strcmp("-r", argv[i]) == 0) {
				seed = atoi(argv[++i]);
			}
			else if (strcmp("-T", argv[i]) == 0) {
				targetCols = atoi(argv[++i]);
			}
			else if (strcmp("-v", argv[i]) == 0) {
				verbose++;
			}
			else {
				ifstream inStream;
				inStream.open(argv[i]);
				if (inStream.fail()) logAndExitNow("Unrecognised file/argument in command line. Exiting\n");
				cout << "Hill Climbing Algorithm using <" << argv[i] << ">\n\n";
				readInputFile(inStream, numNodes, numEdges);
				inStream.close();
			}
		}
	}
	catch (...) {
		logAndExitNow("Error in command line arguments. Exiting\n");
	}
	srand(seed);

	if (targetCols < 2 || targetCols > numNodes) targetCols = 2;

	//Start the checks Count
	numConfChecks = 0;

	//Now set up some output files
	ofstream groupTimeStream, groupConfStream;
	groupTimeStream.open("teffort.txt"); groupConfStream.open("ceffort.txt");
	if (groupTimeStream.fail() || groupConfStream.fail()) logAndExitNow("ERROR opening output FILE\n");

	//Do a check to see if we have the empty graph. If so, end immediately.
	if (numEdges <= 0) {
		groupConfStream << "1\t0\n0\tX\t0\n";
		groupTimeStream << "1\t0\n0\tX\t0\n";
		groupConfStream.close();
		groupTimeStream.close();
		logAndExitNow("Graph has no edges. Optimal solution is obviously using one colour. Exiting\n");
	}

	//Start the timer
	clock_t runStart = clock();

	//-------------------------------------------------------------------------------------------
	//Now form an initial solution called candSol. tempSol is used when rebuilding with greedy. I.e. we flip back and forth between the two.
	vector< vector<int> > candSol, tempSol, unplaced;
	vector<int> groupsToRemove, colNode(numNodes);
	makeInitSolution(numNodes, candSol, tempSol, colNode);
	int currentNumGroups = candSol.size();

	//Write details of initial solutions to the files
	if (verbose >= 1) cout << " COLS     CPU-TIME\tCHECKS" << endl;
	duration = (int)(((clock() - runStart) / double(CLOCKS_PER_SEC)) * 1000);
	if (verbose >= 1) cout << setw(5) << currentNumGroups << setw(11) << duration << "ms\t" << numConfChecks << " (via constructive)" << endl;
	groupConfStream << currentNumGroups << "\t" << numConfChecks << "\n";
	groupTimeStream << currentNumGroups << "\t" << duration << "\n";


	//	MAIN ALGORITHM
	i = 0;
	while (numConfChecks < maxConfChecks && currentNumGroups > targetCols) {

		applySearch(candSol, 1 / double(currentNumGroups), itLimit, tempSol, groupsToRemove, unplaced, colNode);

		if (candSol.size() < currentNumGroups) {
			currentNumGroups = candSol.size();
			duration = (int)(((clock() - runStart) / double(CLOCKS_PER_SEC)) * 1000);
			if (verbose >= 1) cout << setw(5) << currentNumGroups << setw(11) << duration << "ms\t" << numConfChecks << endl;
			groupConfStream << currentNumGroups << "\t" << numConfChecks << "\n";
			groupTimeStream << currentNumGroups << "\t" << duration << "\n";
		}

		if (verbose >= 2)cout << "          -> Iteration " << i << endl;

		i++;
	}
	// END OF MAIN ALGORITHM

	duration = (int)(((clock() - runStart) / double(CLOCKS_PER_SEC)) * 1000);
	if (currentNumGroups <= targetCols) {
		if (verbose >= 1) cout << "\nSolution with  <=" << targetCols << " colours has been found. Ending..." << endl;
		groupConfStream << "1\t" << "X" << "\n";
		groupTimeStream << "1\t" << "X" << "\n";
		ofstream resultsLog("resultsLog.log", ios::app);
		resultsLog << "HC\t" << targetCols << "\t" << (int)(((clock() - runStart) / double(CLOCKS_PER_SEC)) * 1000) << "\t" << numConfChecks << endl;
		resultsLog.close();
	}
	else {
		if (verbose >= 1) cout << "\nRun limit exceeded. No solution using " << currentNumGroups - 1 << " colours was achieved (Checks = " << numConfChecks << ", " << duration << "ms)" << endl;
		groupConfStream << currentNumGroups - 1 << "\t" << "X\t" << numConfChecks << "\n";
		groupTimeStream << currentNumGroups - 1 << "\t" << "X\t" << (int)(((clock() - runStart) / double(CLOCKS_PER_SEC)) * 1000) << "\n";
		ofstream resultsLog("resultsLog.log", ios::app);
		resultsLog << "HC\t" << currentNumGroups << "\t" << (int)(((clock() - runStart) / double(CLOCKS_PER_SEC)) * 1000) << "\t" << numConfChecks << endl;
		resultsLog.close();
	}

	//output the solution to a text file
	ofstream solStrm;
	solStrm.open("solution.txt");
	vector<int> grp(numNodes);
	for (i = 0; i < candSol.size(); i++) { for (int j = 0; j < candSol[i].size(); j++) { grp[candSol[i][j]] = i; } }
	solStrm << numNodes << "\n";
	for (i = 0; i < numNodes; i++) solStrm << grp[i] << "\n";
	solStrm.close();

	//Finally, delete any arrays etc and close output streams
	groupTimeStream.close();
	groupConfStream.close();

}
