#include "tabu.h"
#include "manipulateArrays.h"

#include <iostream>
#include <stdlib.h>

extern unsigned long long numConfChecks;

using namespace std;

int tabu(Graph& g, vector<int>& c, int k, int maxIterations, int verbose, int** neighbors)
{
	int** nodesByColor; // Arrays of nodes for each color
	int* nbcPosition;   // Position of each node in the above array
	int** conflicts;   // Number of conflicts for each color and node
	int** tabuStatus;  // Tabu status for each node and color
	int* nodesInConflict = new int[g.n + 1];
	int* confPosition = new int[g.n];

	long totalIterations = 0;
	int incVerbose = 0;
	int totalConflicts = 0;

	if (verbose == 1) incVerbose = 10000;
	if (verbose == 2) incVerbose = 100;
	if (verbose > 2) incVerbose = 1;
	int nextVerbose = incVerbose;

	int tabuTenure = 5; //This is effetively a random choice

	initializeArrays(nodesByColor, conflicts, tabuStatus, nbcPosition, g, c, k);
	// Count the number of conflicts and set up the list nodesInConflict
	// with the associated list confPosition
	nodesInConflict[0] = 0;
	for (int i = 0; i < g.n; i++) {
		numConfChecks++;
		if (conflicts[c[i]][i] > 0) {
			totalConflicts += conflicts[c[i]][i];
			nodesInConflict[(confPosition[i] = ++nodesInConflict[0])] = i;
		}
	}
	totalConflicts /= 2;
	if (verbose > 1) cout << "Initialized the arrays. #Conflicts = " << totalConflicts << endl;

	int bestSolutionValue = totalConflicts; // Number of conflicts

	// Just in case we already have an admissible k-coloring
	if (bestSolutionValue == 0) {
		return 0;
	}

	int minSolutionValue = g.n;
	int maxSolutionValue = 0;

	//Main TABU LOOP
	while (totalIterations < maxIterations) {
		totalIterations++;
		int nc = nodesInConflict[0];

		int bestNode = -1, bestColor = -1, bestValue = g.n * g.n;
		int numBest = 0;

		// Try for every node in conflict
		for (int iNode = 1; iNode <= nodesInConflict[0]; iNode++) {
			int node = nodesInConflict[iNode];
			// to move it to every color except its existing one
			for (int color = 1; color <= k; color++) {
				if (color != c[node]) {
					numConfChecks += 2;
					int newValue = totalConflicts + conflicts[color][node] - conflicts[c[node]][node];
					if (newValue <= bestValue && color != c[node]) {
						if (newValue < bestValue) {
							numBest = 0;
						}
						// Only consider the move if it is not tabu or leads to a new very best solution seen globally.
						if (tabuStatus[node][color] < totalIterations || (newValue < bestSolutionValue)) {
							// Select the nth move with probability 1/n
							if ((rand() % (numBest + 1)) == 0) {//r.getInt(0,numBest)==0) {
								//we will move node "bestNode" to the new colour "bestColour"
								bestNode = node;
								bestColor = color;
								bestValue = newValue;
							}
							numBest++;  // Count the number of considered moves
						}
					}
				}
			}
		}

		// If no non tabu moves have been found, take any random move
		if (bestNode == -1) {
			bestNode = rand() % g.n;
			while ((bestColor = (rand() % k) + 1) != c[bestNode]); {
				bestValue = totalConflicts + conflicts[bestColor][bestNode] - conflicts[c[bestNode]][bestNode];
				numConfChecks += 2;
			}
		}

		// Now execute the move
		if (verbose > 2) {
			cout << "Will move node " << bestNode << " to color " << bestColor << " with value " << bestValue << " oldconf = " << conflicts[c[bestNode]][bestNode] << " newconf = " << conflicts[bestColor][bestNode] << " totalConflicts = " << totalConflicts << endl;
		}

		int tTenure = tabuTenure;
		moveNodeToColorForTabu(bestNode, bestColor, g, c, nodesByColor, conflicts, nbcPosition, neighbors, nodesInConflict, confPosition, tabuStatus, totalIterations, tTenure);
		totalConflicts = bestValue;

		//Now update the tabu tenure
		tabuTenure = (int)(0.6 * nc) + (rand() % 10);

		// check: have we a new globally best solution?
		if (totalConflicts < bestSolutionValue) {
			bestSolutionValue = totalConflicts;

			// If all nodes are colored we report success and stop iterating
			if (bestSolutionValue == 0) {
				//We have found a feasible solution with k colours, so we jump out of the tabu loop
				break;
			}
			// Otherwise reinitialize some values
			minSolutionValue = g.n * g.n;
			maxSolutionValue = 0;
			nextVerbose = totalIterations;
		}

		//Do some output if needed
		if (totalIterations == nextVerbose && incVerbose) {
			cout << totalIterations << "   obj =" << totalConflicts << "   best =" << bestSolutionValue << "   tenure =" << tabuTenure << endl;
			nextVerbose += incVerbose;
		}

	}// END OF TABU LOOP

	freeArrays(nodesByColor, conflicts, tabuStatus, nbcPosition, k, g.n);
	delete[] nodesInConflict;
	delete[] confPosition;


	return totalConflicts;

}
