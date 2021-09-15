#include "inputGraph.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <string>
#include <stdlib.h>

using namespace std;

void logAndExit(string s) {
	//Writes message s to screen and log file and then exits the program
	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << s;
	cout << s;
	resultsLog.close();
	exit(1);
}

void inputDimacsGraph(Graph& g, char* file) {
	char c, str[400];
	ifstream IN;
	IN.open(file);
	if (IN.fail()) logAndExit("Unrecognised inputfile/argument. Exiting\n");
	int line = 0;
	g.nbEdges = 0;
	int edges = -1;
	try {
		while (!IN.eof()) {
			line++;
			IN.get(c);
			if (IN.eof()) break;
			switch (c) {
			case 'p':
				IN.get(c);
				IN.getline(str, 39, ' ');
				if (strcmp(str, "edge") && strcmp(str, "edges")) {
					logAndExit("Error on line " + to_string(line) + ". No 'edge' keyword found. Exiting\n");
				}
				IN >> g.n;
				IN >> edges;
				g.resize(g.n);
				break;
			case 'e':
				int node1, node2;
				IN >> node1 >> node2;
				if (node1 < 1 || node1 > g.n || node2 < 1 || node2 > g.n) {
					logAndExit("Error on line " + to_string(line) + ". Node label out of range. Exiting\n");
				}
				node1--;
				node2--;
				if (g[node1][node2] == 0) {
					g.nbEdges++;
				}
				else {
					logAndExit("Error on line " + to_string(line) + ". Multiple edge defined. Exiting\n");
				}
				g[node1][node2] = 1;
				g[node2][node1] = 1;
				break;
			case 'c':
				IN.putback('c');
				IN.get(str, 399, '\n');
				break;
			default:
				logAndExit("INVALID FILE. Unrecognised character on line " + to_string(line) + ". Exiting.\n");
			}
			IN.get(); // Kill the newline;
		}
	}
	catch (...) {
		logAndExit("Error reading input file. Exiting.\n");
	}
	IN.close();
}
