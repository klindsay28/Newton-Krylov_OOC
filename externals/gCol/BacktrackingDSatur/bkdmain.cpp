/* This C++ source code of a DSatur-based backtracking algorithm for graph colouring has been adapted from the
	C code of Joseph Culberson and Dennis Papp which is available from Joseph Culberson's Coloring Page
	http://web.cs.ualberta.ca/~joe/Coloring/lclindex.html (Copyright (c) 1997 Joseph Culberson. All rights reserved.)

	Points to note:
	* The program accepts as input a DIMACS formatted graph colouring file
	* Other optional command line settings can be viewed by running the program from the
		command line with no arguments
	* Some of the parameters in this implementation have been fixed. See the publication:

	Lewis, R. (2015) A Guide to Graph Colouring: Algorithms and Applications. Berlin, Springer.
	   ISBN: 978-3-319-25728-0. http://www.springer.com/us/book/978331925728

	for further details
*/

#include "mysys.h"
#include "bktdsat.h"
#include "graph.h"
#include "colorrtns.h"
#include <fstream>
#include <limits.h>

//This makes sure the compiler uses _strtoui64(x, y, z) with Microsoft Compilers, otherwise strtoull(x, y, z) is used
#ifdef _MSC_VER
#define strtoull(x, y, z) _strtoui64(x, y, z)
#endif

//Global Variables
unsigned long long numConfChecks;
unsigned long long maxChecks;
int verbose;
ofstream timeStream, checksStream;
clock_t startTime;
int finalCols;

void colorsearch(char* name, int targetnumcolors)
{
	popmembertype m;
	int i, branch, min, max;
	colortype targetcolor;
	char info[256];

	for (i = 0; i < order; i++) m.vc[i].vertex = i;
	computedeg();
	qsort((char*)m.vc, (int)order, sizeof(struct vrtxandclr), (compfunc)decdeg);

	//These paramters are fixed in this version (in Culberson's original C code they can be altered)
	targetcolor = targetnumcolors;

	//Entering a branching factor of 0 causes the algorithm to behave like DSATUR, essentially performing a sequential search.
	//For larger values, when the program backtracks to some point and takes an alternate branch, the branching factor is
	//reduced by one for the entire subtree. If the branch factor at the root of a subtree is 0 then no further branching is allowed in the subtree.
	//If the branching factor is set very large, and no backtracking region is excluded (see below) then this
	//algorithm guarantees an optimal coloring for all graphs (given excess time)
	branch = INT_MAX;

	//For some graphs improvements might only occur when branching is permitted at certain leaves of the search tree.
	//Entering a pair of numbers such as min=30, max=280 on a graph of 300 vertices means that no branching will occur at depths in that range.
	//Here, min=max=1 meaning branching is permitted at all levels
	min = 1;
	max = 1;

	//Start the timer,
	startTime = clock();

	//And go to the backtracking algorithm itself
	bktdsat(&m, branch, targetcolor, min, max);
	getcolorinfo(&m);

	//Print the final lines to the output files for consistency
	timeStream << "1\tX\n";
	checksStream << "1\tX\n";

	//verify things and end
	verifycolor(&m);
	fileres(name, &m, info);
}

void logAndExitNow(string s) {
	//Writes message s to screen and log file and then exits the program
	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << s;
	cout << s;
	resultsLog.close();
	exit(1);
}

int main(int argc, char* argv[]) {

	if (argc <= 1) {
		cout << "Backtracking DSatur Algorithm for Graph Colouring\n\n"
			<< "USAGE:\n"
			<< "<InputFile>     (Required. File must be in DIMACS format)\n"
			<< "-s <int>        (Stopping criteria expressed as number of constraint checks. Can be anything up to 9x10^18. DEFAULT = 100,000,000.)\n"
			<< "-r <int>        (Random seed. DEFAULT = 1)\n"
			<< "-T <int>        (Target number of colours. Algorithm halts if this is reached. DEFAULT = 1.)\n"
			<< "-v              (Verbosity. If present, output is sent to screen. If -v is repeated, more output is given.)\n"
			<< "****\n";
		exit(1);
	}

	//The following variables are used for keeping track of the number of constraint checks
	numConfChecks = 0;

	//Read in the graph and runtime parameters and/or set default values
	char filename[200];
	int seed = 1, targetNumCols = 2, i;
	verbose = 0;
	maxChecks = 100000000;
	try {
		for (i = 1; i < argc; i++) {
			if (strcmp("-T", argv[i]) == 0) {
				targetNumCols = atoi(argv[++i]);
			}
			else if (strcmp("-r", argv[i]) == 0) {
				seed = atoi(argv[++i]);
			}
			else if (strcmp("-s", argv[i]) == 0) {
				maxChecks = strtoull(argv[++i], NULL, 10);
			}
			else if (strcmp("-v", argv[i]) == 0) {
				verbose++;
			}
			else {
				strcpy(filename, argv[i]);
				cout << "Backtracking DSatur Algorithm using <" << filename << ">\n\n";
			}
		}
	}
	catch (...) {
		logAndExitNow("Error in command line arguments. Exiting.\n");
	}

	srand(seed);
	getgraph(filename);

	//Open the output streams
	timeStream.open("teffort.txt"); checksStream.open("ceffort.txt");
	if (timeStream.fail() || checksStream.fail()) {
		logAndExitNow("ERROR opening output file\n");
	}

	//Produce some output
	if (verbose >= 1) cout << " COLS     CPU-TIME(ms)\tCHECKS" << endl;

	//This is the main algorithm bit
	startTime = clock();

	colorsearch(filename, targetNumCols);

	checksStream.close();
	timeStream.close();

	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << "Btr\t" << finalCols << "\t" << (int)((double)(clock() - startTime) / CLOCKS_PER_SEC * 1000) << "\t" << numConfChecks << "\n";
	resultsLog.close();

	return(0);
}
