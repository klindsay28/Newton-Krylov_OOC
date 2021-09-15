#include "diversity.h"

double solDist(vector<int>& Sol1, vector<int>& Sol2, int numCols)
{
	//This is an expensive operation and should only be used for analysis purposes.
	//Both solutions should be complete (though not necesarrily proper)
	int i, j, l, numItems = 0;
	long unionPairs = 0, intersectionPairs = 0;
	double dist1;

	//Make a group-based representation on Sol1 and Sol2, called S and T resp.
	vector<vector<int> > S(numCols, vector<int>()), T(numCols, vector<int>());
	for (i = 0; i < Sol1.size(); i++) {
		S[Sol1[i] - 1].push_back(i);
		T[Sol2[i] - 1].push_back(i);
	}
	for (i = 0; i < S.size(); i++) numItems += S[i].size();

	vector<vector<bool> > X(numItems, vector<bool>(numItems, false));
	for (i = 0; i < S.size(); i++) {
		for (j = 0; j < S[i].size() - 1; j++) {
			for (l = j + 1; l < S[i].size(); l++) {
				X[S[i][j]][S[i][l]] = true;
				X[S[i][l]][S[i][j]] = true;
				unionPairs++;
			}
		}
	}
	for (i = 0; i < T.size(); i++) {
		for (j = 0; j < T[i].size() - 1; j++) {
			for (l = j + 1; l < T[i].size(); l++) {
				if (X[T[i][j]][T[i][l]])
					intersectionPairs++;
				else
					unionPairs++;
			}
		}
	}
	dist1 = (unionPairs - intersectionPairs) / double(unionPairs);
	return(dist1);
}

double measureDiversity(vector<vector<int> >& population, int numCols)
{
	//Measures the distance between all pairs of solutions in the population. The average is the diversity
	int i, j, n = population.size();
	double total = 0.0;
	for (i = 0; i < n - 1; i++) {
		for (j = i + 1; j < n; j++) {
			total += solDist(population[i], population[j], numCols);
		}
	}
	return total / double((n * (n - 1)) / 2);
}
