#include "makesolution.h"

#include <stdlib.h>

extern unsigned long long numConfChecks;

void removeElAtPos(vector<int>& A, int x) {
	//Removes the element at position x (we can only use this when the element ordering is unimportant)
	A[x] = A.back();
	A.pop_back();
}
void removeAdjNodes(Graph& g, vector<int>& Y, int v) {
	//Removes all nodes in Y that are adjacent to v, or that ARE v. Note that no ordering is preserved
	int i = 0;
	while (i < Y.size()) {
		numConfChecks++;
		if (g[Y[i]][v] || Y[i] == v)
			removeElAtPos(Y, i);
		else
			i++;
	}
}
void updateX(vector<vector<int> >& neighbours, vector<int>& X, int v) {
	//Updates X by 1) Marking element v as having been coloured;
	//2) Since v has been coloured, any adjacent uncoloured nodes have their degree reduced by one
	int i;
	X[v] = INT_MIN;
	numConfChecks++;
	for (i = 0; i < neighbours[v].size(); i++) {
		numConfChecks++;
		if (X[neighbours[v][i]] != INT_MIN) X[neighbours[v][i]]--;
	}
}
int rouletteWheel(vector<double>& tauEta, double tauEtaTotal) {
	if (tauEta.size() == 1) return(0);
	if (tauEtaTotal == 0) return(rand() % tauEta.size());
	//Performs a roulette wheel spin and return the position that we land on
	int i;
	//generate a psuedo-random double between 0.0 (inclusive) and tauEtaTotal (not inclusive)
	double rTotal = 0.0;
	double r = rand() / (double(RAND_MAX) + 1);
	r = r * tauEtaTotal;
	//Now identify the correct "sector position"
	for (i = 0; i < tauEta.size(); i++) {
		if (rTotal <= r && r < (rTotal + tauEta[i])) return i;
		else rTotal += tauEta[i];
	}
	//If we are here then we may have had a choice such as:
	//{0, 1.4822e-323, 0, 0} with tauEtaTotal = 1.4822e-323 and r = 1.4822e-323
	//The value 1.4822e-323 is as close to zero as we can get (without being zero)
	//and could be a result of multiple reductions in t due to evaporation. However, it suggests that
	//eta is not equal to zero, so its in our interests to select it if possible
	for (i = 0; i < tauEta.size(); i++) if (tauEta[i] > 0.0) return i;
	//This captures any other problems, by choosing uniform randomly
	return(rand() % tauEta.size());
}
int chooseMinEdges(vector<vector<int> >& tempX) {
	int i, j, totaldeg, min = INT_MAX, minpos = 0;
	for (i = 0; i < tempX.size(); i++) {
		totaldeg = 0;
		for (j = 0; j < tempX[i].size(); j++) {
			numConfChecks++;
			if (tempX[i][j] >= 0)
				totaldeg += tempX[i][j];
		}
		if (totaldeg < min) {
			min = totaldeg;
			minpos = i;
		}
	}
	return(minpos);
}
//-------------------------------------------------------------------------------------
bool buildSolution(Graph& g, vector<vector<int> >& S, vector<vector<int> >& neighbours, vector<int>& degree, vector<vector<double> >& t, int k, double alpha,
	double beta, int numISets, vector<int>& X, vector<int>& Y, vector< vector<int> >& tempX, vector< vector<int> >& tempY, vector< vector<int> >& ISet,
	vector<double>& tauEta)
{
	int i, j, u, v, iset, col = 0, r, numColoured = 0;
	double tauEtaTotal, tau, eta;
	bool complete = true;

	//X keeps track of ALL nodes that have and have not been assigned. It also holds the degrees of the
	//subgraph induced by X. When a node is assigned to a colour, its degree is set to INT_MIN
	//(Thus in this case Y is a subset of X -- they are not disjoint!)
	X = degree;

	while (numColoured < g.n) {

		//Y holds all vertices that are not coloured and which are suitable for the current colour
		//Determine the set of currently uncoloured nodes Y (these are ones whose degrees are not INT_MIN)
		Y.clear();
		for (i = 0; i < g.n; i++) if (X[i] >= 0) Y.push_back(i);

		//Make "numISets" copies of X and Y and make placeholders for the "numISets" independent sets we are about to produce
		for (i = 0; i < numISets; i++) {
			tempX[i] = X;
			tempY[i] = Y;
			ISet[i].clear();
		}

		for (iset = 0; iset < numISets; iset++) {

			//Choose the first node at random from tempY[iset]. Add to ISet[iset]. "Remove" from tempX[iset] and remove all adjacents from tempY[iset]
			r = rand() % tempY[iset].size();
			v = tempY[iset][r];
			ISet[iset].push_back(v);
			removeAdjNodes(g, tempY[iset], v);
			updateX(neighbours, tempX[iset], v);

			//Choose the remaining nodes due to a) pheremone (measured by tau) and degrees of nodes in the current subgraph induced by tempX[iset]
			while (!tempY[iset].empty()) {
				tauEta.clear();
				tauEta.resize(tempY[iset].size(), 0.0);
				tauEtaTotal = 0.0;
				for (i = 0; i < tempY[iset].size(); i++) {
					//Calculate tau value
					tau = 0.0;
					for (j = 0; j < ISet[iset].size(); j++) {
						tau += t[ISet[iset][j]][tempY[iset][i]];
					}
					tau = tau / double(ISet[iset].size());
					tau = pow(tau, alpha);
					//Calculate eta value
					numConfChecks++;
					eta = double(tempX[iset][tempY[iset][i]]);
					eta = pow(eta, beta);
					//Combine the two
					tauEta[i] = tau * eta;
					tauEtaTotal += tauEta[i];
				}

				//Now select element in tempY[iset] (called u) according to roulette wheel
				r = rouletteWheel(tauEta, tauEtaTotal);
				u = tempY[iset][r];
				ISet[iset].push_back(u);
				removeAdjNodes(g, tempY[iset], u);
				updateX(neighbours, tempX[iset], u);
			}
		}

		//We have now formed "numISets" independent sets. Choose the one that results in the least number of edges in
		//the remaining uncoloured vertices
		r = chooseMinEdges(tempX);

		//Assign these vertices to S[col]
		numColoured += ISet[r].size();
		S[col].swap(ISet[r]);
		X.swap(tempX[r]);

		col++;
		if (col == k && numColoured < g.n) {
			complete = false;
			break;
		}
	}

	//Have produced a (possibly partial) k-colouring. It may also contain empty colour classes, so these are removed
	i = 0;
	while (i < S.size()) {
		if (S[i].empty()) {
			S.back().swap(S[i]);
			S.pop_back();
		}
		else {
			i++;
		}
	}

	if (!complete) {
		//Solution is incomplete. Assign uncoloured nodes at random and return false
		for (i = 0; i < g.n; i++) {
			if (X[i] >= 0) {
				//Node i is currently uncoloured, so assign it to a random colour;
				S[rand() % S.size()].push_back(i);
			}
		}
		return false;
	}
	else {
		//A feasible solution has been produced
		return true;
	}
}

//-------------------------------------------------------------------------------------
void prettyPrintSolution(vector< vector<int> >& candSol)
{
	int i, count = 0, group;
	cout << "\n\n";
	for (group = 0; group < candSol.size(); group++) {
		cout << "Group " << group << " = {";
		if (candSol[group].size() == 0) cout << "empty}\n";
		else {
			for (i = 0; i < candSol[group].size() - 1; i++) {
				cout << candSol[group][i] << ", ";
			}
			cout << candSol[group][candSol[group].size() - 1] << "} |G| = " << candSol[group].size() << "\n";
			count = count + candSol[group].size();
		}
	}
	cout << "Total Number of Nodes = " << count << endl;
}

//---------------------------------------------------------------
void checkSolution(vector< vector<int> >& candSol, Graph& g)
{
	int j, i, count = 0, group;
	bool valid = true;

	//first check that the permutation is the right length
	for (group = 0; group < candSol.size(); group++) {
		count = count + candSol[group].size();
	}

	if (count != g.n) {
		cout << "Error: Permutations length is not equal to the problem size\n";
		valid = false;
	}

	//Now check that all the nodes are in the permutation once
	vector<int> a(g.n, 0);
	for (group = 0; group < candSol.size(); group++) {
		for (i = 0; i < candSol[group].size(); i++) {
			a[candSol[group][i]]++;
		}
	}
	for (i = 0; i < g.n; i++) {
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
					if (g[candSol[group][i]][candSol[group][j]]) {
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
