#include "xover.h"
#include "hungarian.h"
#include "Kempe.h"

extern unsigned long long numConfChecks;

inline
void swap(int& a, int& b) {
	int temp; temp = a; a = b; b = temp;
}
inline
void chooseParents(int popSize, vector<int>& parents)
{
	int i, r;
	//Make a random permutation of numbers 0 to popSize-1
	vector<int> A(popSize);
	for (i = 0; i < popSize; i++) A[i] = i;
	for (i = A.size() - 1; i >= 0; i--) {
		r = rand() % (i + 1);
		swap(A[i], A[r]);
	}
	//The first of these are the randomly selected parents
	for (i = 0; i < parents.size(); i++) parents[i] = A[i];
}
inline
int maxVal(int a, int b) {
	if (a > b)return a; else return b;
}
inline
int intersection(vector<int>& A, vector<int>& B)
{
	int i = 0, j = 0, inter = 0;
	while (i < A.size() && j < B.size()) {
		if (A[i] == B[j]) {
			inter++; i++; j++;
		}
		else if (A[i] > B[j]) j++;
		else i++;
	}
	return(inter);
}
inline
void removeConflicts(vector<int>& sol, Graph& g)
{
	int i, j, r;
	//Make a random permutation of the vertices
	vector<int> A(g.n);
	for (i = 0; i < g.n; i++) A[i] = i;
	for (i = (g.n) - 1; i >= 0; i--) {
		r = rand() % (i + 1);
		swap(A[i], A[r]);
	}
	//Use this to go through all pairs and remove conflicts
	for (i = 0; i < g.n - 1; i++) {
		for (j = i + 1; j < g.n; j++) {
			numConfChecks++;
			if (sol[A[i]] == sol[A[j]] && g[A[i]][A[j]]) {
				//Vertices causing a clash. Remove vertex i
				sol[A[i]] = INT_MIN;
				break;
			}
		}
	}
}

/*********************************************************************************************************************************/
// Procedures for MultiParentXover. If numParents is set to 2 then we get GPX crossover in its original form
/*********************************************************************************************************************************/
inline
void copyColour(int x, int y, int col, vector<vector<int> >& parentCpys, vector<int>& osp, vector<vector<int> >& parentCard, Graph& g)
{
	//This copies all nodes with colour x in parent y into osp, labelling them as colour col.
	int i, j;
	for (i = 0; i < g.n; i++) {
		if (parentCpys[y][i] == x) {
			osp[i] = col;
			for (j = 0; j < parentCpys.size(); j++) {
				if (parentCpys[j][i] != INT_MIN) {
					parentCard[j][parentCpys[j][i]]--;
				}
				parentCpys[j][i] = INT_MIN;
			}
		}
	}
}
int chooseBiggestCol(vector<vector<int> >& parentCard, vector<int>& tabuList, int& x, int& y, int currentCol, int k)
{
	//Choose entry in the rows of the array that are available with the biggest value (break ties randomly)
	//x refers to the colour chosen, y the parent it is in.
	int i, p, maxCard = -100;
	int numParents = tabuList.size();
	vector<int> xList, yList;

	for (p = 0; p < numParents; p++) {
		if (tabuList[p] < currentCol) {
			//Parent p can be considered for colour transfer
			for (i = 1; i <= k; i++) {
				if (parentCard[p][i] > 0) {
					if (parentCard[p][i] > maxCard) {
						xList.clear();		yList.clear();
						xList.push_back(i);	yList.push_back(p);
						maxCard = parentCard[p][i];
					}
					else if (parentCard[p][i] == maxCard) {
						xList.push_back(i);	yList.push_back(p);
					}
				}
			}
		}
	}
	if (xList.empty()) {
		//No choices of colour available
		return(-1);
	}
	else {
		//We now have a list of valid choices for the next colour class. Choose one at random;
		i = rand() % xList.size();
		x = xList[i];
		y = yList[i];
		return(1);
	}
}
inline
void multiParent(vector<int>& osp, vector<int>& parents, Graph& g, int k, vector<vector<int> >& population, bool doKempeMutation)
{
	int i, col, x, y, numUnplaced = 0, j;
	int numParents = parents.size();
	int Q = numParents / 2;

	//Keeps a record at for when parents can be considered for colour transfer. Initially all parents are available
	vector<int>	tabuList(numParents, 0);

	//Initialise the offspring array.
	for (i = 0; i < g.n; i++) osp[i] = INT_MIN;

	//Now copy the parents, and calculate their colour cardinality arrays (remember cols go from 1 up to k inclusive)
	vector<vector<int> > parentCpys(numParents, vector<int>(g.n));
	vector<vector<int> > parentCard(numParents, vector<int>(k + 1, 0));
	for (i = 0; i < numParents; i++) {
		if (!doKempeMutation) {
			//Just make copies of the parents from the population
			for (j = 0; j < g.n; j++) {
				parentCpys[i][j] = population[parents[i]][j];
				parentCard[i][parentCpys[i][j]]++;
			}
		}
		else {
			//Need to strip out conflicts before making the colour cardinality array;
			parentCpys[i] = population[parents[i]];
			removeConflicts(parentCpys[i], g);
			for (j = 0; j < g.n; j++)
				if (parentCpys[i][j] != INT_MIN) parentCard[i][parentCpys[i][j]]++;
		}
	}

	//Now perform the crossover
	for (col = 1; col <= k; col++) {
		//We build up colour col. First choose the biggest colour x from an available parent y (break ties randomly)
		chooseBiggestCol(parentCard, tabuList, x, y, col, k);
		//Update tabuList. Parent y cannot be considered for the next Q colours
		tabuList[y] = col + Q;
		//Copy colour across to the offspring osp (assuming a colour was found) and remove these vertices from all parents
		if (x != -1) copyColour(x, y, col, parentCpys, osp, parentCard, g);
	}

	if (doKempeMutation) {
		//osp is a proper, partial solution we intend to peturb
		doRandomPeturbation(osp, k, g);
	}

	//Assign any remaining uncoloured nodes randomly
	for (i = 0; i < g.n; i++) {
		if (osp[i] == INT_MIN) {
			osp[i] = (rand() % k) + 1;
			numUnplaced++;
		}
	}
}

/*********************************************************************************************************************************/
// Procedures for GGA Crossover
/*********************************************************************************************************************************/
void matchP1toP2(vector<vector<int> >& p1Copy, vector<vector<int> >& p2Copy, int k)
{
	int i, j, maxVal = INT_MIN;
	vector<vector<int> > matrix(k, vector<int>(k)), temp(k, vector<int>());
	vector<int> matching;
	//Calculate intersection size between each pair of groups from diff parents
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			matrix[i][j] = intersection(p1Copy[i], p2Copy[j]);
			if (matrix[i][j] > maxVal) maxVal = matrix[i][j];
		}
	}
	//Make the matrix a minimisation matrix and then find the optimal matching via hungarian
	for (i = 0; i < k; i++) for (j = 0; j < k; j++) matrix[i][j] = maxVal - matrix[i][j];
	doHungarian(k, matrix, matching);
	//Relabel the groups in p1Copy to maxMatch those of p2Copy
	for (i = 0; i < k; i++) p1Copy[i].swap(temp[matching[i]]);
	p1Copy.swap(temp);
}
inline
void randomlyPermuteGroups(vector<vector<int> >& sol)
{
	int i, ran;
	for (i = sol.size() - 1; i >= 0; i--) {
		ran = rand() % (i + 1);
		//now we swap the two groups
		sol[i].swap(sol[ran]);
	}
}
void GGA(vector<int>& osp, vector<int>& parents, Graph& g, int k, vector<vector<int> >& population)
{
	int i, j, x1, x2, numUnplaced = 0, numInjected = 0, p1 = parents[0], p2 = parents[1];
	//Make group representation copy of P1 and p2
	vector<vector<int> > p1Copy(k, vector<int>()), p2Copy(k, vector<int>()), temp(k, vector<int>());
	//Do this by simply copying the information directly from the parents in the population
	for (i = 0; i < g.n; i++) {
		p1Copy[population[p1][i] - 1].push_back(i);
		p2Copy[population[p2][i] - 1].push_back(i);
	}
	//Randomly relabel groups in p2
	randomlyPermuteGroups(p2Copy);
	//Relabel groups in p1 to maximise sum of group intersections with p2
	matchP1toP2(p1Copy, p2Copy, k);
	//Choose 2 crossoverpoints x1<=x2. All groups between x1 (inclusive),...,x2 (not inclusive) inclusive will come from p2;
	x1 = rand() % (k);
	x2 = rand() % (k + 1);
	if (x1 > x2)swap(x1, x2);

	//Now copy details into osp. First, set all valuse in osp to a minus value
	for (i = 0; i < g.n; i++)osp[i] = INT_MIN;
	//Now copy in information from p1Copy
	for (i = 0; i < x1; i++) {
		for (j = 0; j < p1Copy[i].size(); j++) osp[p1Copy[i][j]] = i + 1;
	}
	for (i = x2; i < k; i++) {
		for (j = 0; j < p1Copy[i].size(); j++)	osp[p1Copy[i][j]] = i + 1;
	}
	//Now copy in information from p2Copy, possibly overwriting stuff that came from p1
	for (i = x1; i < x2; i++) {
		for (j = 0; j < p2Copy[i].size(); j++)	osp[p2Copy[i][j]] = i + 1;
		numInjected += p2Copy[i].size();
	}

	//Finally, assign any remaining uncoloured nodes randomly
	for (i = 0; i < g.n; i++) {
		if (osp[i] == INT_MIN) {
			osp[i] = (rand() % k) + 1;
			numUnplaced++;
		}
	}
}

/*********************************************************************************************************************************/
// Procedures for nPoint Crossover
/*********************************************************************************************************************************/
inline
void relabelMaxMatch(vector<int>& p1, vector<int>& p2, Graph& g, int k)
{
	int i, j, maxVal = INT_MIN;
	//Square matrix to hold sizes of each group intersection
	vector<vector<int> > matrix(k, vector<int>(k));
	vector<int> matching;
	//Make group representation copy of P1 and p2
	vector<vector<int> > p1Copy(k, vector<int>()), p2Copy(k, vector<int>());
	for (i = 0; i < g.n; i++) {
		p1Copy[p1[i] - 1].push_back(i);
		p2Copy[p2[i] - 1].push_back(i);
	}
	//Calculate intersection size between each pair of groups from diff parents
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			matrix[i][j] = intersection(p1Copy[i], p2Copy[j]);
			if (matrix[i][j] > maxVal) maxVal = matrix[i][j];
		}
	}
	//Make the matrix a minimisation matrix and then find the optimal matching via hungarian
	for (i = 0; i < k; i++) for (j = 0; j < k; j++) matrix[i][j] = maxVal - matrix[i][j];
	doHungarian(k, matrix, matching);
	//Relabel the groups in p1 to maxMatch those of p2 (remember, groups are labelled 1,...,k in population)
	for (i = 0; i < g.n; i++) p1[i] = matching[p1[i] - 1] + 1;
}
inline
void nPointX(vector<int>& osp, vector<int>& parents, Graph& g, int k, vector<vector<int> >& population)
{
	double xrate = 0.5;
	int i, r, mSize = 0, p1 = parents[0], p2 = parents[1];
	//Relabel groups in p1 so that they overlap as much as possible with p2
	relabelMaxMatch(population[p1], population[p2], g, k);
	//Make offspring via xover and mutation
	for (i = 0; i < g.n; i++) {
		if (rand() / double(RAND_MAX) <= xrate)
			osp[i] = population[p1][i];
		else
			osp[i] = population[p2][i];
		//Mutate with probability 1/n
		if (rand() % g.n == i) {
			do {
				r = (rand() % k) + 1;
			} while (r == osp[i]);
			osp[i] = r;
			mSize++;
		}
	}
}

/*********************************************************************************************************************************/
void doCrossover(int xOverType, vector<int>& osp, vector<int>& parents, Graph& g, int k, vector<vector<int> >& population)
{
	chooseParents(population.size(), parents);
	if (xOverType == 1)		multiParent(osp, parents, g, k, population, false);
	else if (xOverType == 2)	multiParent(osp, parents, g, k, population, true);
	else if (xOverType == 3)	multiParent(osp, parents, g, k, population, false);
	else if (xOverType == 4)	GGA(osp, parents, g, k, population);
	else					nPointX(osp, parents, g, k, population);
}
/*********************************************************************************************************************************/
