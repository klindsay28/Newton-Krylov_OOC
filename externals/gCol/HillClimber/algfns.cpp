#include "algfns.h"
#include <limits.h>
#include <algorithm>

//CONSTANTS USED FOR IDENTIFYING KEMP CHAINS
const int WHITE = 0;
const int GREY = 1;
const int BLACK = 2;

extern unsigned long long numConfChecks;
extern vector< vector<bool> > adjacent;
extern vector< vector<int> > adjList;
extern vector<int> degree;

void logAndExitPlease(string s) {
	//Writes message s to screen and log file and then exits the program
	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << s;
	cout << s;
	resultsLog.close();
	exit(1);
}

//-------------------------------------------------------------------------------------
inline
void RandomlyPermuteASection(vector<int>& permutation, int left, int right)
{
	//This procedure permutes a vector using 2 values: left and right.
	//Say we have the following permutation (length 10) and left = 1 and right = 5
	//_____________________
	//|0|1|2|3|4|5|6|7|8|9|
	//   ^       ^
	//The procedure will then permute the numbers 1,2,3,and 4 ONLY. I.e. "right" will not
	//be touched.
	//We can also permute the entire array by using left = 0 and right = permutation.size()
	if (!permutation.empty()) {
		int i, r, temp;
		for (i = right - 1; i >= left; i--) {
			r = left + rand() % ((i - left) + 1);
			temp = permutation[i]; 	permutation[i] = permutation[r]; permutation[r] = temp;
		}
	}
}

//-------------------------------------------------------------------------------------
inline
void randomlyPermuteTheGroups(vector< vector<int> >& candSol)
{
	int i, ran;
	for (i = candSol.size() - 1; i >= 0; i--) {
		ran = rand() % (i + 1);
		//now we swap the two groups
		candSol[i].swap(candSol[ran]);
	}
}

//-------------------------------------------------------------------------------------
inline
void sortGroupsDescending(vector< vector<int> >& A)
{
	//Because there are many different sorted orderings (because lots of classes might
	//have the same size, first the groups are randomly permuted then sorted
	randomlyPermuteTheGroups(A);
	sort(A.begin(), A.end(), [](const vector<int>& a, const vector<int>& b) { return a.size() > b.size(); });
}

//-------------------------------------------------------------------------------------
inline
void sortGroupsReverse(vector< vector<int> >& A)
{
	int left = 0, right = A.size() - 1;
	while (left < right) {
		//the vectors are to be swapped
		A[left].swap(A[right]);
		left++;
		right--;
	}
}

//-------------------------------------------------------------------------------------
inline
void randomlyPermuteWithinEachGroup(vector< vector<int> >& candSol)
{
	int i;
	for (i = 0; i < candSol.size(); i++) {
		RandomlyPermuteASection(candSol[i], 0, candSol[i].size());
	}
}

//-------------------------------------------------------------------------------------
inline
void removeAnyEmptyGroups(vector< vector<int> >& candSol)
{
	int i = 0;
	while (i < candSol.size()) {
		if (candSol[i].empty())
			candSol.erase(candSol.begin() + i);
		else
			i++;
	}
}

//-------------------------------------------------------------------------------------
inline
void greedyCol(vector< vector<int> >& oldSol, vector< vector<int> >& newSol, vector<int>& colNode)
{
	//This procedure takes a pre-grouped candidate solution and applies the greedy colouring
	//algorithm. The new solution is copied into a new vector and this is returned
	int i;

	//First we delete any items in newSol. This will be built up into the new solution
	for (i = 0; i < newSol.size(); i++) {
		newSol[i].clear();
	}

	//Reinitialise colNode vector (stores the colour of each node, INT_MIN if not assigned)
	for (i = 0; i < colNode.size(); i++) colNode[i] = INT_MIN;

	//Copy the first group into the new solution
	for (i = 0; i < oldSol[0].size(); i++) {
		newSol[0].push_back(oldSol[0][i]);
		colNode[oldSol[0][i]] = 0;
	}

	//Now we go through the remaining groups in the oldSolution, we take each node in turn and see if
	//it is suitable for any existing colour. If it isn't, we shove it in the end colour (we don't need to
	//check if this is suitable
	int oldgroup, newgroup, theItem;
	bool inserted;
	for (oldgroup = 1; oldgroup < oldSol.size(); oldgroup++) {
		for (i = 0; i < oldSol[oldgroup].size(); i++) {
			theItem = oldSol[oldgroup][i];
			//We now check all of the groups 0 to the oldGroup-1 to see if it can be inserted here
			//if it can, we do; if it cannot we simply shove it into newSol[oldGroup-1];
			newgroup = 0;
			inserted = false;
			while (newgroup < oldgroup && !inserted) {
				if (groupIsSuitable(theItem, newSol, newgroup, colNode)) {
					//"theItem" can be inserted into this group. So we do
					newSol[newgroup].push_back(theItem);
					colNode[theItem] = newgroup;
					inserted = true;
				}
				else newgroup++;
			}
			if (!inserted) {
				//If we are here then the item could not be inserted into any of the groups below "oldgroup"
				//so we just stick it into the group it came from (no need to check feasibility)
				newSol[oldgroup].push_back(theItem);
				colNode[theItem] = oldgroup;
			}
		}
	}
	removeAnyEmptyGroups(newSol);

	//IF I WANT TO WORK OUT THE DISTANCE OF THE BEFORE- AND AFTER-SOLUTIONS, THIS IS WHERE I CAN DO IT.
	//solDist(newSol,oldSol);

	//Finally we rename the newly constructed newSol as the oldSolution, which we will use from here on
	newSol.swap(oldSol);
}

//-------------------------------------------------------------------------------------
inline
void selectGroupsToRemove(vector<int>& groupsToRemove, int currentNumGroups, double removalRate)
{
	//This procedure selects which groups to remove from the permutation and stores these in
	//"groupsToRemove"
	int i;
	double ra;

	if (currentNumGroups == 3) {
		//Simply select one group from the three
		groupsToRemove.push_back(rand() % 3);
	}
	else {
		for (i = 0; i < currentNumGroups; i++) {
			ra = rand() / double(RAND_MAX);
			if (ra < removalRate) {
				//we have chosen to remove group i from the permutation
				//cout<<"have chosen to remove group "<<i<<" from the permutation\n";
				groupsToRemove.push_back(i);
			}
		}
	}

	//If no groups have been chosen in the previous loop then this is undesirable, so we simply select one anyway
	if (groupsToRemove.size() == 0) {
		i = rand() % currentNumGroups;
		groupsToRemove.push_back(i);
	}

	//We now need to do an error check because it is possible (particularly for instances with large groups)
	//that TOO MANY of the groups will have been selected for removal. In order to cope with this situation, if
	//less than 2 groups are due to remain, then we select some others to remain as well
	int numGroupsNotSelected = (currentNumGroups - groupsToRemove.size());
	if (numGroupsNotSelected == 0) {
		//randomly select two different groups that were selected for removal and change the decision
		int rand1, rand2;
		rand1 = rand() % groupsToRemove.size();
		groupsToRemove.erase(groupsToRemove.begin() + rand1);
		rand2 = rand() % groupsToRemove.size();
		groupsToRemove.erase(groupsToRemove.begin() + rand2);
	}
	else if (numGroupsNotSelected == 1) {
		//randomly select one of the groups that was selected for removal and change the decision
		groupsToRemove.erase(groupsToRemove.begin() + rand() % groupsToRemove.size());
	}
}


//-------------------------------------------------------------------------------------
void applySearch(vector< vector<int> >& candSol, double removalRate, int lsLimit, vector< vector<int> >& tempSol,
	vector<int>& groupsToRemove, vector< vector<int> >& unplaced, vector<int>& colNode)
{
	int j, i, currentNumGroups = candSol.size();

	if (currentNumGroups <= 2) {
		logAndExitPlease("Two or less groups are being used. This will cause the program to Crash. Aborting...\n");
	}

	//Check to see if we are to apply the Heuristic Search Routine or not
	if (lsLimit > 0) {
		//If we are here then we want to apply the HS procedure.
		//First, we want to select some "groups" for removal according to the removalRate. Groups that are to
		//be reomved are put into the vector "groupsToRemove"
		groupsToRemove.clear();
		selectGroupsToRemove(groupsToRemove, currentNumGroups, removalRate);

		//We now copy all of the the items that we plan to remove from the candidate solution into a second 2d vector
		//called "unplaced". We then empty the groups that we wanted to remove, and remove these empty groups
		unplaced.clear();
		for (i = 0; i < groupsToRemove.size(); i++) {
			int remGroup = groupsToRemove[i];
			unplaced.push_back(vector<int>());
			for (j = 0; j < candSol[remGroup].size(); j++) {
				unplaced[i].push_back(candSol[remGroup][j]);
			}
			candSol[remGroup].clear();
		}
		removeAnyEmptyGroups(candSol);

		//Next, before passing candSol and unplaced to the HS procedure we first randomly shuffle the groups in unplaced.
		randomlyPermuteTheGroups(unplaced);

		//Problem Specific: For flexibility, we also permute the items WITHIN each group
		randomlyPermuteWithinEachGroup(candSol);
		randomlyPermuteWithinEachGroup(unplaced);

		//We are now ready to apply the LS procedure
		runLS(candSol, unplaced, lsLimit);

		//Once the Iteration Limit of the HS procedure has been reached, we are finally able to rebuild into a
		//full solution. To do this, unplaced is appended onto the candidate solution
		candSol.insert(candSol.end(), unplaced.begin(), unplaced.end());
	}

	//When we get here (whether having run HS or not) we should have a full solution again. We therefore
	//simply rerun the greedyAlgorithm (recolour)
	rebuild(candSol, tempSol, colNode);
}

//-------------------------------------------------------------------------------------
inline
void runLS(vector< vector<int> >& candSol, vector< vector<int> >& U, int itLimit)
{
	int group1 = 0, group2 = 0, g1, g2, myRand;
	bool doneAMove = false, elimColour = false;
	int groupsToCheck = candSol.size();
	int count = 0;
	vector<int> randPerm(candSol.size()), temp0, temp1;
	vector< vector<int> > theColours(2);
	vector< vector<int> > theGroups(2);
	vector<int> groupsToConsider(2);

	while ((!U.empty() && (count < itLimit))) {

		//If we are in the 1st iteration of local search we check against all groups; otherwise we only check the recently altered groups
		if (groupsToCheck == candSol.size())
			force(U, candSol);
		else { //groupsToCheck == 2
			groupsToConsider[0] = group1;
			groupsToConsider[1] = group2;
			force(U, candSol, groupsToConsider);
		}

		//SHUFFLE BIT. We now want to alter the partial solution candSol
		if (!U.empty()) {
			doneAMove = false;
			do {
				//If we are here then we are to alter the solution in some way, ensuring that feasibility is maintained
				myRand = rand() % 100;
				if (myRand == 99) {
					//The swap item operator is applied
					performItemSwap(g1, g2, candSol, randPerm, doneAMove);
					groupsToCheck = 2;
				}
				else {
					//The Kempe operator is applied
					performKempeChainInterchange(candSol, g1, g2, theColours, theGroups, temp0, temp1, elimColour, doneAMove);
					groupsToCheck = 2;
					if (elimColour) {
						//A colour has been eliminated by the Kempe Chain interchange, so we jump out of the local search loop
						return;
					}
				}
				count++;
			} while (!doneAMove && count < itLimit);
			group1 = g1;
			group2 = g2;
		}
		else {
			//If we are here, we have moved all vertices from U so we want to end
			return;
		}
	}
}


//-------------------------------------------------------------------------------------
inline
void performItemSwap(int& firstGroup, int& secGroup, vector< vector<int> >& candSol, vector<int>& randPerm, bool& doneAMove)
{
	//First, we set up a random permutation of the groups
	int i, i1, i2, j;
	for (i = 0; i < candSol.size(); i++) randPerm[i] = i;
	RandomlyPermuteASection(randPerm, 0, candSol.size());

	//Go through each pair of non-adjacent nodes in different groups in a random way until we find a pair that can be swapped. Then swap them and end
	//Only non adjacent node swaps are considered as the others can be achieved using the Kempe Chain operator
	for (i = 0; i < randPerm.size() - 1; i++) {
		firstGroup = randPerm[i];
		for (i1 = 0; i1 < candSol[firstGroup].size(); i1++) {
			for (j = i + 1; j < randPerm.size(); j++) {
				secGroup = randPerm[j];
				for (i2 = 0; i2 < candSol[secGroup].size(); i2++) {
					numConfChecks++;
					if (!adjacent[candSol[firstGroup][i1]][candSol[secGroup][i2]]) {
						if (neighbourhoodSwapIsFeasible(candSol, firstGroup, i1, secGroup, i2)) {
							//The swapping of candSol[firstGroup][i1] and candSol[secondGroup][i2] is OK, so we do it
							int temp = candSol[firstGroup][i1];
							candSol[firstGroup][i1] = candSol[secGroup][i2];
							candSol[secGroup][i2] = temp;
							//We have achieved our goal and performed the swap so we return true and end
							doneAMove = true;
							return;
						}
					}
				}
			}
		}
	}
	doneAMove = false;
	return;
}

//-------------------------------------------------------------------------------------
inline
void performKempeChainInterchange(vector< vector<int> >& candSol, int& group0, int& group1,
	vector< vector<int> >& theColours, vector< vector<int> >& theGroups, vector<int>& temp0, vector<int>& temp1, bool& elimColour, bool& doneAMove)
{
	int i, blackCount = 0, nodePos;

	//Select a node randomly
	group0 = rand() % candSol.size();
	nodePos = rand() % candSol[group0].size();

	//Select a second colour that is different to the first
	do {
		group1 = rand() % candSol.size();
	} while (group0 == group1);

	//Set up some data structures for use with the restricted DepthFirstSearch Routine we're about to do
	theColours[0].clear();
	theColours[1].clear();
	theGroups[0].clear();
	theGroups[1].clear();
	for (i = 0; i < candSol[group0].size(); i++) {
		theColours[0].push_back(WHITE);
		theGroups[0].push_back(candSol[group0][i]);
	}
	for (i = 0; i < candSol[group1].size(); i++) {
		theColours[1].push_back(WHITE);
		theGroups[1].push_back(candSol[group1][i]);
	}

	//Now start the restricted DFS from the chosen node
	DFSVisit(nodePos, 0, theColours, theGroups, blackCount);

	//When we get to here we are able to identify which nodes are in the Kempe chain by seeing which nodes are BLACK.
	//If all the nodes in the 2 colour classes are BLACK then swapping would do nothing more
	//than a colour-relabelling. This is no good for our purposes so we return false
	int numNodesInTheTwoGroups = theGroups[0].size() + theGroups[1].size();
	if (blackCount == numNodesInTheTwoGroups) {
		//This would only be a colour relabelling
		elimColour = false;
		doneAMove = false;
		return;
	}
	else if (blackCount == 1 && candSol[group0].size() == 1) {
		//This implies that the Selected Kempe chain interchange will result in a reduction in the number of colours
		//(i.e. a colour class with a single vertex, will have this moved to a different colour class)
		swapNodesInKempeChain(theColours, candSol, theGroups, group0, group1, temp0, temp1);
		elimColour = true;
		doneAMove = true;
		removeAnyEmptyGroups(candSol);
		return;
	}
	else {
		//If we are here then we going to do the swap, but the number of colours remains the same
		swapNodesInKempeChain(theColours, candSol, theGroups, group0, group1, temp0, temp1);
		elimColour = false;
		doneAMove = true;
		return;
	}
}

//-------------------------------------------------------------------------------------
inline
void DFSVisit(int uPos, int theGroup, vector< vector<int> >& theColours, vector< vector<int> >& theGroups, int& blackCount)
{
	int theOtherGroup, vPos;
	if (theGroup == 0) theOtherGroup = 1;
	else theOtherGroup = 0;

	theColours[theGroup][uPos] = GREY;

	for (vPos = 0; vPos < theColours[theOtherGroup].size(); vPos++) {
		numConfChecks++;
		if (adjacent[theGroups[theGroup][uPos]][theGroups[theOtherGroup][vPos]]) {
			if (theColours[theOtherGroup][vPos] == WHITE) {
				DFSVisit(vPos, theOtherGroup, theColours, theGroups, blackCount);
			}
		}
	}
	theColours[theGroup][uPos] = BLACK;
	blackCount++;
}

//-------------------------------------------------------------------------------------
inline
void swapNodesInKempeChain(vector< vector<int> >& theColours, vector< vector<int> >& candSol, vector< vector<int> >& theGroups,
	int group0, int group1, vector<int>& temp0, vector<int>& temp1)
{
	//Having Identified the Kempe chain, this operator now swaps the colors of the relevant nodes in
	//the two colour classes. It does this by making 2 new vectors and replacing the old ones
	int i;
	temp0.clear();
	temp1.clear();
	//Put into temp0 all of the nodes that were in
	for (i = 0; i < theGroups[0].size(); i++) {
		if (theColours[0][i] == BLACK)
			//this node is to be moved into Group1
			temp1.push_back(theGroups[0][i]);
		else
			//This node is to stay in group 0
			temp0.push_back(theGroups[0][i]);
	}
	for (i = 0; i < theGroups[1].size(); i++) {
		if (theColours[1][i] == BLACK)
			//this node is to be moved into Group0
			temp0.push_back(theGroups[1][i]);
		else
			//This node is to stay in group 0
			temp1.push_back(theGroups[1][i]);
	}
	//Now replace the relevant parts of the candidate solution with these new vectors
	candSol[group0].swap(temp0);
	candSol[group1].swap(temp1);
}

//-------------------------------------------------------------------------------------
inline
bool neighbourhoodSwapIsFeasible(vector< vector<int> >& candSol, int g1, int i1, int g2, int i2)
{
	int i;
	int item1 = candSol[g1][i1];
	int item2 = candSol[g2][i2];

	//Here we want to check to see it item1 can be put into group2 and item2 into group 1

	//First, check item1's suitability in g2;
	for (i = 0; i < i2; i++) {
		numConfChecks++;
		if (adjacent[item1][candSol[g2][i]]) return false;
	}
	for (i = i2 + 1; i < candSol[g2].size(); i++) {
		numConfChecks++;
		if (adjacent[item1][candSol[g2][i]]) return false;
	}

	//Next, check item2's suitability in g1
	for (i = 0; i < i1; i++) {
		numConfChecks++;
		if (adjacent[item2][candSol[g1][i]]) return false;
	}
	for (i = i1 + 1; i < candSol[g1].size(); i++) {
		numConfChecks++;
		if (adjacent[item2][candSol[g1][i]]) return false;
	}

	//if we are here then swap is OK
	return(true);
}

//-------------------------------------------------------------------------------------
inline
bool groupIsSuitable(int v, vector< vector<int> >& sol, int c, vector<int>& colNode)
{
	//This checks to see whether vertex v can be feasibly inserted into colour c in sol. It can be
	//done in one of 2 ways - go through each neighbour of v, or go through all vertices in colour c
	int i;
	numConfChecks++;
	if (sol[c].size() > degree[v]) {
		//check if any neighbours of v are currently in colour c
		for (i = 0; i < adjList[v].size(); i++) {
			numConfChecks++;
			if (colNode[adjList[v][i]] == c) return false;
		}
		return true;
	}
	else {
		//check if any vertices in colour c are adjacent to v
		for (i = 0; i < sol[c].size(); i++) {
			numConfChecks++;
			if (adjacent[v][sol[c][i]]) return false;
		}
		return true;
	}
}
inline
bool groupIsSuitable(int v, vector<int>& col)
{
	//Overloaded version of above
	int i;
	for (i = 0; i < col.size(); i++) {
		numConfChecks++;
		if (adjacent[v][col[i]]) return false;
	}
	return true;
}

//-------------------------------------------------------------------------------------
inline
void eraseElementFromIntVector(vector<int>& theVec, int thePos)
{
	//This function removes the element at position "thePos" in element from "theVec".
	//We use this as an alternative to "vector.erase()" which operates in linear time.
	//NB: it can only be used when the "ordering" of the items in the vector is unimportant
	theVec[thePos] = theVec[theVec.size() - 1];
	theVec.pop_back();
}


//-------------------------------------------------------------------------------------
inline
void force(vector< vector<int> >& U, vector< vector<int> >& candSol)
{
	//This version of "force" goes through every item in U and checks every group
	//in candSol to see if it can be inserted there
	int i, solgroup, ugroup;
	bool doneInsertion;

	ugroup = 0;
	while (ugroup < U.size() && !U.empty()) {
		i = 0;
		while (i < U[ugroup].size() && !U[ugroup].empty()) {
			doneInsertion = false;
			solgroup = 0;
			while (solgroup < candSol.size() && !doneInsertion) {
				//Check to see if item U[ugroup][i] can go into group "solGroup" in the candidate solutiom
				//(this is problem specific)
				if (groupIsSuitable(U[ugroup][i], candSol[solgroup])) {
					//cout << "I can put " << U[ugroup][i] << " into group "<<solgroup<<endl;
					candSol[solgroup].push_back(U[ugroup][i]);
					doneInsertion = true;
				}
				else solgroup++;
			}
			if (doneInsertion)
				eraseElementFromIntVector(U[ugroup], i);
			//U[ugroup].erase(U[ugroup].begin() + i);
			else i++;
		}
		if (U[ugroup].empty()) {
			//if one of the groups in U has become empty then we delete it
			//cout<<"Have emptied a group in U"<<endl;
			U.erase(U.begin() + ugroup);
		}
		else ugroup++;
	}
}

//-------------------------------------------------------------------------------------
inline
void force(vector< vector<int> >& U, vector< vector<int> >& candSol, vector<int>& groupsToConsider)
{
	//In this version of "force" we will only try to shove the items into the 2 groups that have
	//been altered ince last time. For convienience we put the 2 groups we are considering into a
	//vector
	int i, considerCnt, ugroup;
	bool doneInsertion;

	//Now it is just the same as before (more or less)
	ugroup = 0;
	while (ugroup < U.size() && !U.empty()) {
		i = 0;
		while (i < U[ugroup].size() && !U[ugroup].empty()) {
			doneInsertion = false;
			considerCnt = 0;
			while (considerCnt < groupsToConsider.size() && !doneInsertion) {
				//Check to see if item U[ugroup][i] can go into group "groupsToConsider[considerCnt]" in the
				//candidate solutiom (this is problem specific)
				if (groupIsSuitable(U[ugroup][i], candSol[groupsToConsider[considerCnt]])) {
					candSol[groupsToConsider[considerCnt]].push_back(U[ugroup][i]);
					doneInsertion = true;
				}
				else considerCnt++;
			}
			if (doneInsertion) {
				eraseElementFromIntVector(U[ugroup], i);
			}
			else i++;
		}
		if (U[ugroup].empty()) {
			//if one of the groups in U has become empty then we delete it
			U.erase(U.begin() + ugroup);
		}
		else ugroup++;
	}
}

//-------------------------------------------------------------------------------------
inline
void rebuild(vector< vector<int> >& candSol, vector< vector<int> >& tempSol, vector<int>& colNode)
{
	//First we decide how to permute the groups of this complete solution
	//We do this using the ration from culberson's paper
	int ratioTotal = 130;
	//Here we use the ratio 50:50:30
	int ran = rand() % ratioTotal;
	if (ran < 50) sortGroupsDescending(candSol);
	else if (ran < 100)sortGroupsReverse(candSol);
	else randomlyPermuteTheGroups(candSol);

	//PROBLEM SPECIFIC: for GCol, we can also permute inside each group
	randomlyPermuteWithinEachGroup(candSol);

	//Finally, we can run the greedy algorithm
	greedyCol(candSol, tempSol, colNode);
}
