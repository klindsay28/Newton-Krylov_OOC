#include "display.h"
#include <cstring>
#include <string>

extern vector< vector<bool> > adjacent;
extern vector< vector<int> > adjList;
extern vector<int> degree;

void logAndExit(string s) {
	//Writes message s to screen and log file and then exits the program
	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << s;
	cout << s;
	resultsLog.close();
	exit(1);
}

//-------------------------------------------------------------------------------------
void prettyPrintSolution(vector< vector<int> >& candSol)
{
	int i, count = 0, group;
	cout << "\n\n";
	for (group = 0; group < candSol.size(); group++) {
		cout << "Colour " << group << " = {";
		if (candSol[group].size() == 0) cout << "-}\n";
		else {
			for (i = 0; i < candSol[group].size() - 1; i++) {
				cout << candSol[group][i] << ", ";
			}
			cout << candSol[group][candSol[group].size() - 1] << "}\n";
			count = count + candSol[group].size();
		}
	}
	cout << "Total Number of Nodes = " << count << endl;
}

//---------------------------------------------------------------
void checkSolution(vector< vector<int> >& candSol, int numItems)
{
	int j, i, count = 0, group;
	bool valid = true;

	//first check that the permutation is the right length
	for (group = 0; group < candSol.size(); group++) {
		count = count + candSol[group].size();
	}

	if (count != numItems) {
		cout << "Error: Permutations length is not equal to the problem size\n";
		valid = false;
	}

	//Now check that all the nodes are in the permutation once
	vector<int> a(numItems, 0);
	for (group = 0; group < candSol.size(); group++) {
		for (i = 0; i < candSol[group].size(); i++) {
			a[candSol[group][i]]++;
		}
	}
	for (i = 0; i < numItems; i++) {
		if (a[i] != 1) {
			cout << "Error: Item " << i << " is not present " << a[i] << " times in the solution\n";
			valid = false;
		}
	}

	//Finally, check for illegal colourings: I.e. check that each colour class contains non conflicting nodes
	for (group = 0; group < candSol.size(); group++) {
		if (!candSol[group].empty()) {
			for (i = 0; i < candSol[group].size() - 1; i++) {
				for (j = i + 1; j < candSol[group].size(); j++) {
					if (adjacent[candSol[group][i]][candSol[group][j]]) {
						cout << "Error: Nodes " << candSol[group][i] << " and " << candSol[group][j] << " are in the same group, but they clash" << endl;
						valid = false;
					}
				}
			}
		}
	}
	if (valid) cout << "This solution is valid" << endl;
	else cout << "This solution is not valid" << endl;
}

void readInputFile(ifstream& inStream, int& numNodes, int& numEdges)
{
	//Reads a DIMACS format file and creates the corresponding degree array and adjacency matrix
	char c;
	char str[1000];
	int line = 0, i, j;
	numEdges = 0;
	int edges = -1;
	try {
		while (!inStream.eof()) {
			line++;
			inStream.get(c);
			if (inStream.eof()) break;
			switch (c) {
			case 'p':
				inStream.get(c);
				inStream.getline(str, 999, ' ');
				if (strcmp(str, "edge") && strcmp(str, "edges")) {
					logAndExit("Error. No 'edge' keyword found on a p line (line " + to_string(line) + "). Exiting\n");
				}
				inStream >> numNodes;
				inStream >> numEdges;
				//Set up the 2d adjacency matrix
				adjacent.resize(numNodes, vector<bool>(numNodes));
				for (i = 0; i < numNodes; i++) for (j = 0; j < numNodes; j++) {
					if (i == j)adjacent[i][j] = true;
					else adjacent[i][j] = false;
				}
				break;
			case 'e':
				int node1, node2;
				inStream >> node1 >> node2;
				if (node1 < 1 || node1 > numNodes || node2 < 1 || node2 > numNodes) {
					logAndExit("Error. Invalid note at line " + to_string(line) + ". Exiting\n");
				}
				node1--;
				node2--;
				if (!adjacent[node1][node2]) {
					edges++;
				}
				else {
					logAndExit("Error. Duplicate edge at line " + to_string(line) + ". Exiting\n");
				}
				adjacent[node1][node2] = true;
				adjacent[node2][node1] = true;
				break;
			case 'c':
				inStream.putback('c');
				inStream.get(str, 999, '\n');
				break;
			default:
				logAndExit("INVALID FILE. File line " + to_string(line) + " is invalid. Exiting\n");
			}
			inStream.get(); // Kill the newline;
		}
	}
	catch (...) {
		logAndExit("INVALID FILE. Exiting\n");
	}
	inStream.close();

	//Finally, use the adjacency matrix to construct the degree array
	degree.resize(numNodes, 0);
	adjList.resize(numNodes, vector<int>());
	for (i = 0; i < numNodes; i++) {
		for (j = 0; j < numNodes; j++) {
			if (adjacent[i][j] && i != j) {
				adjList[i].push_back(j);
				degree[i]++;
			}
		}
	}
}
