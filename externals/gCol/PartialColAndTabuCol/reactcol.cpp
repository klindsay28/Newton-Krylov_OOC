#include "reactcol.h"
#include "initializeColoring.h"
#include "manipulateArrays.h"
#include <iostream>
#include <stdlib.h>

using namespace std;

extern unsigned long long numConfChecks;

long reactcol(Graph& g, int* c, int k, unsigned long long maxChecks, int staticTenure, int verbose, int freq, int inc, int** neighbors) {

	int** nodesByColor; // Arrays of nodes for each color
	int* nbcPosition;   // Position of each node in the above array
	int** conflicts;   // Number of conflicts for each color and node
	int** tabuStatus;  // Tabu status for each node and color

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

	if (freq) { // frenquency and increment are set manually
		frequency = freq;
		increment = inc;
	}

	long totalIterations = 0;
	int currentIterations = 0;
	int incVerbose = 0;
	if (verbose == 1) incVerbose = 10000;
	if (verbose == 2) incVerbose = 100;
	if (verbose > 2) incVerbose = 1;
	int nextVerbose = incVerbose;
	int result = -1;

	int tabuTenure = increment;
	int randomTenure = 1;
	if (staticTenure != 0) {
		tabuTenure = staticTenure;
		randomTenure = 0;
	}

	initializeColoring(g, c, k);
	//if (verbose>1) cout << "Initialized the coloring\n";

	initializeArrays(nodesByColor, conflicts, tabuStatus, nbcPosition, g, c, k);
	//if (verbose>1) cout << "Initialized the arrays. |Outnodes| = " << nodesByColor[0][0] << endl;

	int bestSolutionValue = nodesByColor[0][0]; // Number of out nodes

	// Just in case we already have an admissible k-coloring
	if (bestSolutionValue == 0) {
		return 0;
	}

	int minSolutionValue = g.n;
	int maxSolutionValue = 0;

	while (numConfChecks < maxChecks) {

		currentIterations++;
		totalIterations++;

		int bestNode = -1, bestColor = -1, bestValue = g.n;
		int numBest = 0;

		// Try for every uncolored outNode
		for (int iOutNode = 1; iOutNode <= nodesByColor[0][0]; iOutNode++) {
			int outNode = nodesByColor[0][iOutNode];
			// to move it to every color

			for (int color = 1; color <= k; color++) {
				numConfChecks++;
				if (conflicts[color][outNode] <= bestValue) {
					numConfChecks++;
					if (conflicts[color][outNode] < bestValue) {
						numBest = 0;
					}

					// Only consider the move if it is not tabu or leads to a new very best solution seen globally.
					numConfChecks++;
					if (tabuStatus[outNode][color] < totalIterations || (conflicts[color][outNode] == 0 && nodesByColor[0][0] == bestSolutionValue)) {

						// Select the nth move with probability 1/n
						if (rand() % (numBest + 1) == 0) {
							bestNode = outNode;
							bestColor = color;
							numConfChecks++;
							bestValue = conflicts[color][outNode];
						}
						numBest++;  // Count the number of considered moves
					}
				}
			}
		}
		// If no non tabu moves have been found, take any random move
		if (bestNode == -1) {
			bestNode = nodesByColor[0][(rand() % nodesByColor[0][0]) + 1];
			bestColor = (rand() % k) + 1;
			bestValue = conflicts[bestColor][bestNode];
			numConfChecks++;
		}

		int tTenure = tabuTenure;
		if (randomTenure == 1) {
			if (tTenure == 0) tTenure++;
			else tTenure = (rand() % tTenure) + 1;
		}

		// Now execute the move
		moveNodeToColor(bestNode, bestColor, g, c, nodesByColor, conflicts, nbcPosition, neighbors, tabuStatus, totalIterations, tTenure);

		// Update the min and max objective function value
		if (nodesByColor[0][0] > maxSolutionValue) maxSolutionValue = nodesByColor[0][0];
		if (nodesByColor[0][0] < minSolutionValue) minSolutionValue = nodesByColor[0][0];

		int Delta = maxSolutionValue - minSolutionValue;

		if (staticTenure == 0) {

			if (currentIterations % frequency == 0) {
				// Adjust the tabuTenure every frequency iterations
				if (Delta < 2 || tabuTenure == 0) {
					tabuTenure += increment;
					if (pairCycles == nextPair) {
						if (!freq) { // frequency and incrment are not set manually
							int p = rand() % numPairs;
							frequency = pairs[p][0];
							increment = pairs[p][1];
							pairCycles = 0;
							nextPair = pairs[p][2];
						}
						randomTenure = rand() % 2;
					}
					else {
						pairCycles++;
					}
				}
				else if (tabuTenure) {
					tabuTenure--;
				}

				minSolutionValue = g.n;
				maxSolutionValue = 0;

			}
		}
		else {
			tabuTenure = (int)(0.6 * nodesByColor[0][0]) + rand() % 10;
		}

		// Have we a new globally best solution?
		if (nodesByColor[0][0] < bestSolutionValue) {
			bestSolutionValue = nodesByColor[0][0];

			// If all nodes are colored we report success and stop iterating
			if (bestSolutionValue == 0) {
				result = 1;
				break;
			}
			// Otherwise reinitialize some values
			minSolutionValue = g.n;
			maxSolutionValue = 0;
			currentIterations = 0;
			pairCycles = 0;
			nextVerbose = totalIterations;
		}

		if (verbose >= 2 && totalIterations % 1000 == 0)
			cout << "          -> Iteration " << totalIterations << " Cost = " << nodesByColor[0][0] << endl;


		if (totalIterations == nextVerbose && incVerbose) {
			nextVerbose += incVerbose;
		}
	}

	freeArrays(nodesByColor, conflicts, tabuStatus, nbcPosition, k, g.n);

	if (verbose >= 2)cout << "          -> Iteration " << totalIterations << " Cost = " << bestSolutionValue << endl;

	return bestSolutionValue;
}
