#include "mysys.h"

#include <limits.h>
#include <fstream>
#include <float.h>
#include <iomanip>
#include "colorrtns.h"
#include "graph.h"
#include "maxclique.h"
#include "bktdsat.h"

extern unsigned long long numConfChecks;
extern unsigned long long maxChecks;
extern ofstream timeStream, checksStream;
extern int verbose;
extern clock_t startTime;
extern int finalCols;

/*choose minsat */
#define MAXCLR 1280
#define LONG_BITS	BITS(long)
#define MOD_MASK	(LONG_BITS-1)
#define SHIFT_VALUE	5

/* prototypes */
void BlockColor(vertextype v, colortype c, colortype maxclr, int branch, popmembertype* m);
void FindPair(colortype maxclr, vertextype* v, colortype* c, int* impval);
void move(vertextype v, int newsatur);
void fix(void);
void ApplyColor(vertextype v, colortype c, colortype maxclr, int branch, popmembertype* m);
void Color(colortype maxclr, int branch, popmembertype* m);
int impact(vertextype v, colortype c);

/* global variables */
vertextype nextv[MAXCLR + MAXVERTEX];
vertextype Prev[MAXCLR + MAXVERTEX];
vertextype lclindex[MAXCLR + MAXVERTEX];

/* how many colors are conflicting */
vertextype satur[MAXVERTEX];
/* pointer to position in lists */
vertextype current[MAXCLR];

/* total of each adjacent color to vertex */
short clrset[MAXVERTEX][MAXCLR];

colortype bestcolor, maxsat, minsat;
vertextype numcolored;
popmembertype bestmember;
int Fixed;
colortype target;
int maxbranch;
int minlimit, maxlimit;
int MinMax;

/* interrupt control */
int stopflag = 0;
void cpulimbk()
{
	printf("CPU TIME EXCEEDED -- let me clean up\n");
	stopflag = 1;
}
void intfcbk()
{
	printf("INTERRUPT DETECTED -- cleaning up\n");
	stopflag = 1;
}

void bktdsat(popmembertype* m, int branch, colortype targetclr, int min, int max)
{
	/* vertices of equal saturation are kept in a circular doubly linked list. There are MAXCLR such lists. Vertex i is represented 0 <= i <
	MAXVERTEX by the ith position. nextv and Prev indicate the nextv and Previous vertices in the list.
	lclindex indicates the lclindex number of the vertex in the permutation handed to brelaz. Each list is kept in its
	lclindex order. The positions MAXVERTEX <= i < MAXCLR represent the reference positions in the circular lists; lclindex for these
	positions is ENDLIST to make it easy to detect end of list, and by keeping ENDLIST large makes it easy to detect insertion conditions at end of list.
	*/
	vertextype v, w;
	vertextype i, j;
	int cliquesize;
	stopflag = 0;

	//Here are some other parameters that have been hard coded from the original version of Culberson:
	//This means at each iteration a vertex of maximum saturation is chosen (conforming to the Dsatur algorithm)
	MinMax = 1;

	//In this version, we do not attempt to find a large clique, so the following procedure always returns cliquesize = 1
	cliquesize = maxclique(m);
	if (cliquesize > targetclr) {
		printf("WARNING: a clique > target color was found\n");
		target = cliquesize;
	}
	else {
		target = targetclr;
	}
	bestcolor = order + 1;

	//initialise variables
	maxsat = 0;
	minsat = 0;
	numcolored = 0;
	bestmember = *m;
	Fixed = 0;
	maxbranch = order;
	minlimit = min;
	maxlimit = max;

	/* initially no color saturation */
	for (i = 0; i < order; i++) {
		for (j = 0; j < MAXCLR; j++)
			clrset[i][j] = 0;
		satur[i] = 0;
	}

	/* all vertices are on zero conflict list */
	for (i = MAXVERTEX; i < MAXCLR + MAXVERTEX; i++) {
		nextv[i] = Prev[i] = i;
		lclindex[i] = ENDLIST;
	}

	/* the 0th conflict list is anchored in MAXVERTEX position of array */
	w = MAXVERTEX;
	for (i = 0; i < order; i++) {
		lclindex[v = m->vc[i].vertex] = i;
		/* vertices are coming in from smallest to largest, so insert at end of list  thus never changing w.*/
		nextv[v] = w;
		Prev[v] = Prev[w];
		nextv[Prev[w]] = v;
		Prev[w] = v;
	}

	Color(0, branch, m);
	*m = bestmember;
	m->clrdata.numcolors = bestcolor;
}

int impact(vertextype v, colortype c)
{
	adjacencytype* x;
	vertextype w;
	int impval = 0;
	numConfChecks++;
	initnbr(x, v);
	for (w = 0; w < order; w++) {
		numConfChecks++;
		if (isnbr(x, w) && lclindex[w] != ENDLIST && clrset[w][c] == 0)
			impval++;
	}
	return (impval);
}

void fix(void)
{
	int j;

	/* initialize pointers to reference positions */
	for (j = 0; j < MAXCLR; j++) current[j] = j + MAXVERTEX;

	/* scan for maximum saturation list */
	while (nextv[current[maxsat]] == current[maxsat] && maxsat > 0) maxsat--;
	/* scan for min saturation list */
	while (nextv[current[minsat]] == current[minsat] && minsat < maxsat) minsat++;

	Fixed = 1;
}

void Color(colortype maxclr, int branch, popmembertype* m)
{
	/* maxcnf is maxsat */
	vertextype v;
	colortype c;
	int impval;

	if (stopflag) {
		return;
	}

	if (numcolored >= order) {
		if (maxclr < bestcolor) {
			//At this point we have reduced the number of colors by one So we record the time and number of checks
			clock_t colDuration = (int)((double)(clock() - startTime) / CLOCKS_PER_SEC * 1000);
			if (verbose >= 1) cout << setw(5) << maxclr << setw(11) << colDuration << "ms\t" << numConfChecks << endl;
			timeStream << maxclr << "\t" << colDuration << "\n";
			checksStream << maxclr << "\t" << numConfChecks << "\n";
			finalCols = maxclr;
			if (maxclr <= target) {
				if (verbose >= 1) cout << "\nSolution with <=" << maxclr << " colours has been found. Ending..." << endl;
				stopflag = 1;
			}
			bestcolor = maxclr;
			bestmember = *m;
		}
		else {
			if (verbose >= 2) cout << "Backtracking to level " << numcolored - 1 << "..." << endl;
		}
	}
	else if (bestcolor <= target) {
		if (verbose >= 1) cout << "\nSolution with <=" << bestcolor << " colours has been found. Ending..." << endl;
		clock_t colDuration = (int)((double)(clock() - startTime) / CLOCKS_PER_SEC * 1000);
		timeStream << bestcolor << "\t" << colDuration << "\n";
		checksStream << bestcolor << "\t" << numConfChecks << "\n";
		finalCols = bestcolor;
		stopflag = 1;
	}
	else if (maxclr >= bestcolor) {
		if (verbose >= 2) cout << "Worse or equal coloring, backtracking...\n";
	}
	else if (numConfChecks > maxChecks) {
		clock_t colDuration = (int)((double)(clock() - startTime) / CLOCKS_PER_SEC * 1000);
		if (verbose >= 1) cout << "\nRun limit exceeded. No solution using " << bestcolor - 1 << " colours was achieved (Checks = " << numConfChecks << ", " << colDuration << "ms)" << endl;
		timeStream << bestcolor - 1 << "\tX\t" << colDuration << "\n";
		checksStream << bestcolor - 1 << "\tX\t" << numConfChecks << "\n";
		finalCols = bestcolor;
		stopflag = 1;
	}
	else {
		fix();
		if (maxsat == maxclr) {
			/* some vertex is completely saturated */
			v = nextv[current[maxsat]];
			if (maxclr + 1 < bestcolor) {
				if (verbose >= 2) cout << "Level " << numcolored + 1 << ":\tv-" << v << " is fully saturated (sat-deg = " << maxsat << " = maxclr). Assigning to colour " << maxclr + 1 << endl;
				ApplyColor(v, maxclr + 1, maxclr, branch, m);
			}
		}
		else {
			FindPair(maxclr, &v, &c, &impval);
			if (verbose >= 2) cout << "Level " << numcolored + 1 << ":\tAssigning v-" << v << " to colour " << c << " (available colours = " << maxclr <<")" << endl;
			ApplyColor(v, c, maxclr, branch, m);
			if (maxclr >= bestcolor) {
				if (verbose >= 2) cout << "Backtracking to level " << numcolored - 1 << "..." << endl;
				return;
			}
			/* if impact==0 then this color was not at fault */
			if (branch > 0 && impval > 0 && (numcolored<minlimit || numcolored>maxlimit)) {
				if (numcolored <= maxbranch) {
					maxbranch = numcolored;
					fflush(stdout);
				}
				BlockColor(v, c, maxclr, branch - 1, m);
			}
		}
	}
}

/* #### use impact size array to save list of nbrs? */
void ApplyColor(vertextype v, colortype c, colortype maxclr,
	int branch, popmembertype* m)
{
	vertextype oldlclindex, w;
	int oldmaxsat, oldminsat;
	int j;
	adjacencytype* x;

	oldmaxsat = maxsat;
	oldminsat = minsat;

	/* pull v off its list */
	nextv[Prev[v]] = nextv[v];
	Prev[nextv[v]] = Prev[v];

	if (c > maxclr) maxclr = c;

	numcolored++;
	m->vc[lclindex[v]].color = c;
	oldlclindex = lclindex[v];
	lclindex[v] = ENDLIST;	/* no longer on any list */

	/* update saturation and impact lists */
	numConfChecks++;
	initnbr(x, v);
	for (j = 0; j < order; j++) {
		w = m->vc[j].vertex;
		numConfChecks++;
		if (isnbr(x, w) && lclindex[w] != ENDLIST) {
			/* mark color in colorset and check if color not Previously adjacent to w */
			if (0 == (clrset[w][c]++)) {
				/* move vertex to nextv list */
				move(w, satur[w] + 1);
				satur[w]++;
				if (maxsat < satur[w])
					maxsat = satur[w];
			}
		}
	}
	Fixed = 0;
	Color(maxclr, branch, m);

	if (!Fixed) {
		fix();
	}

	/* restore saturation and impact lists */
	for (j = 0; j < order; j++) {
		w = m->vc[j].vertex;
		numConfChecks++;
		if (isnbr(x, w) && lclindex[w] != ENDLIST) {
			/* unmark color in colorset and check if color now not adjacent to w */
			if (0 == (--clrset[w][c])) {
				/* assume satur[w]>0 return vertex to Prev list */
				move(w, satur[w] - 1);
				satur[w]--;
				if (minsat > satur[w])
					minsat = satur[w];
			}
		}
	}
	Fixed = 0;

	/* put v back on its list */
	if (Prev[nextv[v]] != Prev[v])
		printf("ERROR: Prev nextv %d != Prev %d\n", v, v);
	if (nextv[Prev[v]] != nextv[v])
		printf("ERROR: nextv Prev %d != nextv %d\n", v, v);

	Prev[nextv[v]] = v;
	nextv[Prev[v]] = v;

	numcolored--;
	lclindex[v] = oldlclindex;
	m->vc[lclindex[v]].color = 0;

	maxsat = oldmaxsat;
	minsat = oldminsat;
}


void move(vertextype v, int newsatur)
{
	vertextype z, zp;

	nextv[Prev[v]] = nextv[v];
	Prev[nextv[v]] = Prev[v];
	if (current[satur[v]] == v) current[satur[v]] = Prev[v];

	/* insert v into new list. Note use of maximal ENDLIST lclindex */
	z = current[newsatur];
	while (lclindex[zp = nextv[z]] < lclindex[v])
		z = zp;

	nextv[v] = zp;
	Prev[v] = z;
	nextv[z] = v;
	Prev[zp] = v;
	current[newsatur] = v;


}

void FindPair(colortype maxclr, vertextype* v, colortype* c, int* impval)
{
	int w, i, t;
	*impval = order; *c = 1; *v = 0;

	if (MinMax == 0)
		w = nextv[current[minsat]];
	else
		w = nextv[current[maxsat]];

	for (i = 1; i <= maxclr; i++) {
		if (clrset[w][i] == 0) {
			t = impact(w, i);
			if (t < *impval) {
				*impval = t;
				*c = i;
				*v = w;
			}
		}
		if (*impval == 0) break;
	}
}

void BlockColor(vertextype v, colortype c, colortype maxclr, int branch, popmembertype* m)
{
	clrset[v][c] = order;

	fix();
	move(v, satur[v] + 1);
	satur[v]++;
	if (maxsat < satur[v])
		maxsat = satur[v];
	Fixed = 0;

	Color(maxclr, branch, m);
	if (!Fixed) {
		fix();
	}

	move(v, satur[v] - 1);
	satur[v]--;
	if (minsat > satur[v])
		minsat = satur[v];
	Fixed = 0;

	clrset[v][c] = 0;
}
