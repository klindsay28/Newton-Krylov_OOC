#include "hungarian.h"

//-----------STAGE 1-------------------------------------------------------------------------------------------------------
inline
void stage1(vector<vector<cell> >& vals, int n)
{
	int r, c, min;
	//Subtract minimum of each row r
	for (r = 0; r < n; r++) {
		min = INT_MAX;
		for (c = 0; c < n; c++) {
			if (vals[r][c].weight < min) {
				min = vals[r][c].weight;
				if (min == 0) break;
			}
		}
		//Subtract min from each cell in row r;
		if (min > 0) {
			for (c = 0; c < n; c++) vals[r][c].weight -= min;
		}
	}
	//Subtract minimum of each column c
	for (c = 0; c < n; c++) {
		//ID minimum in col c
		min = INT_MAX;
		for (r = 0; r < n; r++) {
			if (vals[r][c].weight < min) {
				min = vals[r][c].weight;
				if (min == 0) break;
			}
		}
		//Subtract min from each cell in col c;
		if (min > 0) {
			for (r = 0; r < n; r++) vals[r][c].weight -= min;
		}
	}
}

//-----------PROCEDURES FOR DRAWING LINES-----------------------------------------------------------------------------------------------
inline
void addVerticalLine(int col, vector<vector<cell> >& vals, int n, vector<bool>& rowLined,
	vector<bool>& colLined, vector<int>& rowZeros, vector<int>& colZeros, int& numZerosRemaining, int& linesAdded)
{
	//Add a vertical line at col (this should not be lined currently. Update the arrays
	int r;
	colLined[col] = true;
	for (r = 0; r < n; r++) {
		if (vals[r][col].weight == 0 && !vals[r][col].lined && vals[r][col].visible) {
			rowZeros[r]--;
			colZeros[col]--;
			numZerosRemaining--;
		}
		vals[r][col].lined = true;
	}
	linesAdded++;
}
inline
void addHorizontalLine(int row, vector<vector<cell> >& vals, int n, vector<bool>& rowLined,
	vector<bool>& colLined, vector<int>& rowZeros, vector<int>& colZeros, int& numZerosRemaining, int& linesAdded)
{
	//Add a vertical line at col (this should not be lined currently. Update the arrays
	int c;
	rowLined[row] = true;
	for (c = 0; c < n; c++) {
		if (vals[row][c].weight == 0 && !vals[row][c].lined && vals[row][c].visible) {
			rowZeros[row]--;
			colZeros[c]--;
			numZerosRemaining--;
		}
		vals[row][c].lined = true;
	}
	linesAdded++;
}
inline
bool checkRows(vector<vector<cell> >& vals, int n, vector<bool>& rowLined, vector<bool>& colLined, vector<int>& rowZeros,
	vector<int>& colZeros, int& numZerosRemaining, int& linesAdded, vector<int>& matching)
{
	//Checks each row to see if it contains the desiredNum of zeros. Usually desiredNum will be 1.
	//However, if no row (or col) contains just one zero, then destiredNum is increased; this
	//procedure then exits as soon as a zero is found and desiredNum is reset to zero
	int r = 0, rowsNoSucc = 0, c;
	bool rowsToCheck = true, haveDoneAlteration = false;
	while (rowsToCheck) {
		if (!rowLined[r] && rowZeros[r] == 1) {
			//Row r contains 1 unmarked zero. ID the column c and add a vertical line there
			for (c = 0; c < n; c++)if (!colLined[c] && vals[r][c].weight == 0 && vals[r][c].visible) break;
			addVerticalLine(c, vals, n, rowLined, colLined, rowZeros, colZeros, numZerosRemaining, linesAdded);
			matching[r] = c;
			haveDoneAlteration = true;
			if (linesAdded == n) return haveDoneAlteration;
			rowsNoSucc = 0;
		}
		else {
			rowsNoSucc++;
			if (rowsNoSucc >= n) break;
		}
		r++;
		if (r == n)r = 0;
	}
	return haveDoneAlteration;
}
inline
bool checkColumns(vector<vector<cell> >& vals, int n, vector<bool>& rowLined, vector<bool>& colLined, vector<int>& rowZeros,
	vector<int>& colZeros, int& numZerosRemaining, int& linesAdded, vector<int>& matching)
{
	//Same as checkRows above, but works with columns
	int r, c = 0, colsNoSucc = 0;
	bool colsToCheck = true, haveDoneAlteration = false;;
	while (colsToCheck) {
		if (!colLined[c] && colZeros[c] == 1) {
			//Col c contains the desired num of unmarked zero. ID a row r and add a horizontal line there
			for (r = 0; r < n; r++)if (!rowLined[r] && vals[r][c].weight == 0 && vals[r][c].visible) break;
			addHorizontalLine(r, vals, n, rowLined, colLined, rowZeros, colZeros, numZerosRemaining, linesAdded);
			matching[r] = c;
			haveDoneAlteration = true;
			if (linesAdded == n) return haveDoneAlteration;
			colsNoSucc = 0;
		}
		else {
			colsNoSucc++;
			if (colsNoSucc >= n) break;
		}
		c++;
		if (c == n)c = 0;
	}
	return haveDoneAlteration;
}
inline
void coverMin(vector<vector<cell> >& vals, int n, vector<bool>& rowLined, vector<bool>& colLined,
	vector<int>& rowZeros, vector<int>& colZeros, int& numZerosRemaining, int& linesAdded,
	vector<int>& matching)
{
	int r, c, i, min = INT_MAX, minPos = -1;
	bool inCol = true;
	//ID the unlined col or row with the min num of uncovered and visible zeros in it >1;
	for (c = 0; c < n; c++) {
		if (!colLined[c] && colZeros[c] < min && colZeros[c] > 0) {
			min = colZeros[c];
			minPos = c;
			inCol = true;
		}
	}
	for (r = 0; r < n; r++) {
		if (!rowLined[r] && rowZeros[r] < min && rowZeros[r] > 0) {
			min = rowZeros[r];
			minPos = r;
			inCol = false;
		}
	}
	if (inCol) {
		//We have IDd a col with >1 zero. Draw a h-line at the first zero
		for (r = 0; r < n; r++)
			if (!rowLined[r] && vals[r][minPos].weight == 0 && vals[r][minPos].visible)
				break;
		addHorizontalLine(r, vals, n, rowLined, colLined, rowZeros, colZeros, numZerosRemaining, linesAdded);
		matching[r] = minPos;
		//Now make the others in the col invisible (they cannot seed a later line-drawing
		for (i = r + 1; i < n; i++) {
			if (!rowLined[i] && vals[i][minPos].weight == 0 && vals[i][minPos].visible) {
				vals[i][minPos].visible = false;
				rowZeros[i]--;
				colZeros[minPos]--;
				numZerosRemaining--;
			}
		}
	}
	else {
		//We have IDd a row with >1 zero. Draw a line at the first zero
		for (c = 0; c < n; c++)if (!colLined[c] && vals[minPos][c].weight == 0 && vals[minPos][c].visible) break;
		addVerticalLine(c, vals, n, rowLined, colLined, rowZeros, colZeros, numZerosRemaining, linesAdded);
		matching[minPos] = c;
		//Now make the others in the row invisible (they cannot seed a later line-drawing
		for (i = c + 1; i < n; i++) {
			if (!colLined[i] && vals[minPos][i].weight == 0 && vals[minPos][i].visible) {
				vals[minPos][i].visible = false;
				colZeros[i]--;
				rowZeros[minPos]--;
				numZerosRemaining--;
			}
		}
	}
}
inline
int drawLines(vector<vector<cell> >& vals, int n, vector<bool>& rowLined,
	vector<bool>& colLined, vector<int>& rowZeros, vector<int>& colZeros, vector<int>& matching)
{
	//Draw the minimum number of lines to cover all zeros. Return the number of lines
	int r, c, i, linesAdded = 0, numZerosRemaining = 0;
	bool doneChangeRow, doneChangeCol;
	//First, set up the arrays, vals, and count the number of zeros
	for (i = 0; i < n; i++) {
		rowLined[i] = false; rowZeros[i] = 0;
		colLined[i] = false; colZeros[i] = 0;
		matching[i] = -1;
	}
	for (r = 0; r < n; r++) {
		for (c = 0; c < n; c++) {
			vals[r][c].visible = true;
			vals[r][c].lined = false;
			if (vals[r][c].weight == 0) {
				rowZeros[r]++;
				colZeros[c]++;
				numZerosRemaining++;
			}
		}
	}
	//Now draw the lines. Halt when there all zeros are covered
	while (numZerosRemaining > 0) {
		doneChangeRow = checkRows(vals, n, rowLined, colLined, rowZeros, colZeros, numZerosRemaining, linesAdded, matching);
		if (linesAdded == n) break;
		doneChangeCol = checkColumns(vals, n, rowLined, colLined, rowZeros, colZeros, numZerosRemaining, linesAdded, matching);
		if (linesAdded == n) break;
		if (!doneChangeRow && !doneChangeCol) {
			coverMin(vals, n, rowLined, colLined, rowZeros, colZeros, numZerosRemaining, linesAdded, matching);
			if (linesAdded == n) break;
		}
	}
	return linesAdded;
}

//-----------PROCEDURES FOR UPDATING THE MATRIX-------------------------------------------------------------------------------------------
inline
void updateVals(vector<vector<cell> >& vals, int n, vector<bool>& rowLined, vector<bool>& colLined)
{
	//ID minimal non-lined number.
	int r, c, min = INT_MAX;
	for (r = 0; r < n; r++) {
		for (c = 0; c < n; c++) {
			if (!vals[r][c].lined && vals[r][c].weight < min && vals[r][c].weight > 0) //Am accepting that there could be zeros not under a line
				min = vals[r][c].weight;
		}
	}
	//Subtract min from all unlined numbers. Add to all numbers at intersection of lines
	for (r = 0; r < n; r++) {
		for (c = 0; c < n; c++) {
			if (!vals[r][c].lined && vals[r][c].weight > 0) {
				//Cell is not lined (or invisible) so subtract min from it.
				vals[r][c].weight -= min;
			}
			else if (rowLined[r] && colLined[c]) {
				//Cell is at intersection of 2 lines so add min on to it
				vals[r][c].weight += min;
			}
		}
	}
}
void doHungarian(int n, vector<vector<int> >& matrix, vector<int>& matching) {
	//PROCEDURE FOR CARRYING OUT THE HUNGARIAN ALGORITHM. RETURNS A VECTOR WITH THE MATCHING IN IT
	//MAIN VARIABLES
	int r, c, lines = 0;
	bool done = false;
	matching.resize(n, -1);

	//Make copy of the original weight matrix
	vector<vector<cell> > vals(n, vector<cell>(n));
	for (r = 0; r < n; r++) for (c = 0; c < n; c++) vals[r][c].weight = matrix[r][c];

	//Declare some other useful things
	vector<bool> rowLined(n), colLined(n);
	vector<int> rowZeros(n), colZeros(n);

	//HERE IS THE HUNGARIAN ALGORITHM
	stage1(vals, n);
	while (!done) {
		lines = drawLines(vals, n, rowLined, colLined, rowZeros, colZeros, matching);
		if (lines < n)
			updateVals(vals, n, rowLined, colLined);
		else break;
	}
}
