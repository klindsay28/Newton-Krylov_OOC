#include "tabu.h"
#include "initializeColoring.h"
#include "manipulateArrays.h"
#include <iostream>
#include <stdlib.h>

using namespace std;

extern unsigned long long numConfChecks;

long tabu(Graph& g, int* c, int k, unsigned long long maxChecks, int staticTenure, int verbose, int freq, int inc, int** neighbors)
{
	int** nodesByColor; // Arrays of nodes for each color
	int* nbcPosition;   // Position of each node in the above array
	int** conflicts;   // Number of conflicts for each color and node
	int** tabuStatus;  // Tabu status for each node and color
	int* nodesInConflict = new int[g.n + 1];
	int* confPosition = new int[g.n];

	//This array is used with the dynamic tenure scheme
	int pairs[][3] = { {10000,10,5},
	{10000,15,3},
	{10000,5,10},
	{5000,15,10},
	{5000,10,15},
	{5000,5,20},
	{1000,15,30},
	{1000,10,50},
	{1000,5,100},
	{500,5,100},
	{500,10,150},
	{500,15,200}
	};

	int numPairs = sizeof(pairs) / sizeof(int) / 3;

	int pairCycles = 0;
	int frequency = pairs[0][0];
	int increment = pairs[0][1];
	int nextPair = pairs[0][2];

	if (freq) {
		frequency = freq;
		increment = inc;
	}

	long totalIterations = 0;
	int currentIterations = 0;
	int incVerbose = 0;
	int totalConflicts = 0;

	if (verbose == 1) incVerbose = 10000;
	if (verbose == 2) incVerbose = 100;
	if (verbose > 2) incVerbose = 1;
	int nextVerbose = incVerbose;
	int result = -1;

	int tabuTenure = g.n / 10;
	int randomTenure = 1;
	if (staticTenure != 0) {
		tabuTenure = staticTenure;
		randomTenure = 0;
	}

	//Make the initial solution
	initializeColoringForTabu(g, c, k);

	//if (verbose>1) cout << "Initialized the coloring\n";

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

	//if (verbose>1) cout << "Initialized the arrays. #Conflicts = " << totalConflicts << endl;

	int bestSolutionValue = totalConflicts; // Number of conflicts

	// Just in case we already have an admissible k-coloring
	if (bestSolutionValue == 0) {
		return 0;
	}

	int minSolutionValue = g.n;
	int maxSolutionValue = 0;

	while (numConfChecks < maxChecks) {

		currentIterations++;
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
							if (rand() % (numBest + 1) == 0) {
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
				numConfChecks += 2;
				bestValue = totalConflicts + conflicts[bestColor][bestNode] - conflicts[c[bestNode]][bestNode];
			}
		}

		// Now execute the move
		if (verbose >= 2 && totalIterations % 1000 == 0)
			cout << "          -> Iteration " << totalIterations << " Cost = " << totalConflicts << endl;

		int tTenure = tabuTenure;
		if (randomTenure == 1) tTenure = (rand() % tTenure) + 1;
		moveNodeToColorForTabu(bestNode, bestColor, g, c, nodesByColor, conflicts, nbcPosition, neighbors, nodesInConflict, confPosition, tabuStatus, totalIterations, tTenure);
		totalConflicts = bestValue;

		int max_min = 0;

		//Now update the tabu tenure
		if (staticTenure == 0) {
			// Update the min and max objective function value
			if (totalConflicts > maxSolutionValue) maxSolutionValue = totalConflicts;
			if (totalConflicts < minSolutionValue) minSolutionValue = totalConflicts;
			max_min = maxSolutionValue - minSolutionValue;
			if (currentIterations % frequency == 0) {
				// Adjust the tabuTenure every frequency iterations
				if (maxSolutionValue - minSolutionValue < 4 + totalConflicts / 80 || tabuTenure <= 1) {
					tabuTenure += increment;
					if (pairCycles == nextPair) {
						if (!freq) {
							// frequency and increment are not set manually
							int p = rand() % numPairs;
							frequency = pairs[p][0];
							increment = pairs[p][1];
							pairCycles = 0;
							nextPair = pairs[p][2];
						}
						randomTenure = rand() % 2;
					}
				}
				else if (tabuTenure) {
					tabuTenure--;
				}

				minSolutionValue = g.n * g.n;
				maxSolutionValue = 0;

				if (pairCycles == nextPair) {
					if (!freq) { // frequency and increment are not set manually
						int p = rand() % numPairs;
						frequency = pairs[p][0];
						increment = pairs[p][1];
						pairCycles = 0;
						nextPair = pairs[p][2];
					}
				}
				else {
					pairCycles++;
				}
			}
		}
		else {
			tabuTenure = (int)(0.6 * nc) + rand() % 10;
		}

		// check: have we a new globally best solution?
		if (totalConflicts < bestSolutionValue) {
			bestSolutionValue = totalConflicts;

			// If all nodes are colored we report success and stop iterating
			if (bestSolutionValue == 0) {
				result = 1;
				break;
			}
			// Otherwise reinitialize some values
			minSolutionValue = g.n * g.n;
			maxSolutionValue = 0;
			currentIterations = 0;
			nextVerbose = totalIterations;
		}

		if (totalIterations == nextVerbose && incVerbose) {
			nextVerbose += incVerbose;
		}

	}// END OF TABU LOOP

	if (verbose >= 2) cout << "          -> Iteration " << totalIterations << " Cost = " << totalConflicts << endl;

	freeArrays(nodesByColor, conflicts, tabuStatus, nbcPosition, k, g.n);
	delete[] nodesInConflict;
	delete[] confPosition;

	return totalConflicts;

}
