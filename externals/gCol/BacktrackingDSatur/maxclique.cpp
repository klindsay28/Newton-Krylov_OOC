#include "mysys.h"
#include <math.h>
#include <stdlib.h>
#include "colorrtns.h"
#include "maxclique.h"

#define MAXIS 500
#define MAXCUT 100
#define isnotnbr(x,i)   !(((x)[(i) >> SHIFT]) & (1 << ((i) & MASK)))

extern unsigned long long numConfChecks;

typedef struct isinfo {
	vertextype	possible, /*vertex not incident to any in IS */
		degree;   /*degree of vertex*/
} istype;


typedef istype twoDarray[MAXIS][MAXVERTEX];
typedef twoDarray* twoDarrayp;

/* globals */
int degree[MAXVERTEX];
int middeg;

int udgcmp(istype* a, istype* b)
/*
Comparison routine for sorting by degree downwards
*/
{
	if (a->degree < b->degree) return(1);
	else if (a->degree == b->degree) return(0);
	else return(-1);
}

int dgcmp(istype* a, istype* b)
/*
Comparison routine for sorting by degree
*/
{
	if (a->degree > b->degree) return(1);
	else if (a->degree == b->degree) return(0);
	else return(-1);
}

int midcmp(istype* a, istype* b)
/*
comparison routine for sorting mid values to front
*/
{
	if (abs(middeg - (int)(a->degree)) > abs(middeg - (int)(b->degree)))
		return(1);
	else if (a->degree == b->degree) return(0);
	else    return(-1);
}

/*
brute force clique finding
using indset() function from maxis - so variable names are
not necessarily appropriate
*/
void clique(
	int	freesize,	/* number of remaining vertices */
	int* retsize,	/* size of best independent set found */
	vertextype bestset[],	/* returned independent set */
	vertextype okaylist[], 	/* list of vertices we start with */
	int	cutlimit[],
	int	cutvertex[],
	int	limit,
	int	usort,
	int	msort)
{
	/*
	A complete rewrite of program for selecting maximum clique
	Data Structure:
	*/
	vertextype curset[MAXIS]; /* current independent set */
	twoDarrayp indsetinf;

	istype* nptr, * pptr;
	adjacencytype* x, * y; /* speed up pointers*/

	int nextv[MAXIS]; /* pointer to next vertex to try in possible */
	int numposs[MAXIS]; /* number in possible */

	int degseq[MAXVERTEX];

	int nextis; /* stack pointer to curset */
	int bestis, bestdeg; /* size counts for best set */
	int lcldeg; /* for computing degree of is */

	int i, j, k; /* for loop controls etc.*/
	int Prev; /* temporary for speed */

	int firsttime; /* depth control */
	int cutoff[MAXCUT];
	int usortlimit, msortlimit;

	int degtot;

	/* coefficients for quadratic cutoff rates */

	indsetinf = (twoDarrayp)malloc(sizeof(twoDarray));
	if (indsetinf == NULL) printf("ERROR: indset: not enough memory.\n");

	/* initialize 0th to the initial set of vertices */
	numposs[0] = freesize;
	pptr = (*indsetinf)[0];
	for (i = 0; i < freesize; i++) {
		pptr[i].possible = okaylist[i];
		pptr[i].degree = 0;
		numConfChecks++;
		initnbr(x, pptr[i].possible);
		for (k = 0; k < i; k++)
			if (isnotnbr(x, (pptr[k].possible))) {
				pptr[i].degree++;
				pptr[k].degree++;
			}
	}
	nextv[0] = 0;

	usortlimit = (freesize * usort) / 100;
	msortlimit = (freesize * msort) / 100;
	k = 0;
	while (cutvertex[k] > numposs[0]) k++;
	if (cutlimit[k] < numposs[0])
		cutoff[0] = cutlimit[k];
	else	cutoff[0] = numposs[0];

	/* set degree sequence */
	degtot = 0;
	for (i = 0; i < freesize; i++) {
		degseq[pptr[i].possible] = pptr[i].degree;
		degtot += pptr[i].degree;
	}

	if (freesize >= 1) middeg = degtot / freesize;
	else middeg = degtot;

	/* sort middles to front */
	if (numposs[0] >= usortlimit) {
		qsort((char*)pptr, (int)numposs[0], sizeof(istype), (compfunc)udgcmp);
	}
	else if (numposs[0] >= msortlimit) {
		qsort((char*)pptr, (int)numposs[0], sizeof(istype), (compfunc)midcmp);
	}
	else {
		qsort((char*)pptr, (int)numposs[0], sizeof(istype), (compfunc)dgcmp);
	}

	bestis = 0;
	bestdeg = 0;
	nextis = 1;
	firsttime = 1;
	while ((nextis > limit) || firsttime) {
		if (nextis >= limit) firsttime = 0;
		/* select next vertex */
		Prev = nextis - 1;
		if (nextv[Prev] >= cutoff[Prev]) {
			/* BACKTRACK */
			nextis--;
		}
		else if (bestis > (Prev + (numposs[Prev] - nextv[Prev]))) {
			/* BOUNDED BACKTRACK  - there are too few vertices
			left to build a better set
			this is most useful on k-colorable Graph
			*/
			nextis--;
		}
		else {
			/* use some speed up variables */
			nptr = (*indsetinf)[nextis];
			pptr = (*indsetinf)[Prev];

			/* select the next vertex */
			curset[nextis] = pptr[nextv[Prev]].possible;
			numConfChecks++;
			initnbr(x, (curset[nextis]));

			/* reset Previous next */
			nextv[Prev]++;

			/* create the possible list */
			nextv[nextis] = 0;
			j = 0;
			/* note: we consider only the remaining vertices
			of Previous possible list */
			for (i = nextv[Prev]; i < numposs[Prev]; i++) {
				if (!(isnotnbr(x, (pptr[i].possible)))) {
					nptr[j].possible = pptr[i].possible;
					nptr[j].degree = 0;
					numConfChecks++;
					initnbr(y, (nptr[j].possible));
					for (k = 0; k < j; k++)
						if (isnotnbr(y, (nptr[k].possible))) {
							nptr[k].degree++;
							nptr[j].degree++;
						}
					j++;
				}
			}
			numposs[nextis] = j;
			degtot = 0; /* mindeg = order; */
			for (i = 0; i < j; i++) {
				degtot += nptr[i].degree;
			}
			if (j > 0) middeg = degtot / j;
			else middeg = degtot;

			if (numposs[nextis] >= usortlimit) {
				qsort((char*)nptr, (int)numposs[nextis], sizeof(istype), (compfunc)udgcmp);
			}
			else if (numposs[nextis] >= msortlimit) {
				qsort((char*)nptr, (int)numposs[nextis], sizeof(istype), (compfunc)midcmp);
			}
			else {
				qsort((char*)nptr, (int)numposs[nextis], sizeof(istype), (compfunc)dgcmp);
			}

			k = 0;
			while (cutvertex[k] > numposs[nextis]) k++;
			if (cutlimit[k] < numposs[nextis])
				cutoff[nextis] = cutlimit[k];
			else cutoff[nextis] = numposs[nextis];

			/* keep track of the best so far */
			if (bestis < nextis) {
				/* copy the set */
				bestdeg = 0;
				for (i = 1; i <= nextis; i++) {
					bestset[i] = curset[i];
					bestdeg += degseq[curset[i]];
				}
				bestis = nextis;
			}
			else if (bestis == nextis) {
				/* compute degree */
				lcldeg = 0;
				for (i = 1; i <= nextis; i++)
					lcldeg += degseq[curset[i]];
				if (bestdeg < lcldeg) {
					for (i = 1; i <= nextis; i++)
						bestset[i] = curset[i];
					bestdeg = lcldeg;
				}
			}

			/* next iteration */
			nextis++;
		}
	}
	*retsize = bestis;
	free(indsetinf);
}

int maxclique(popmembertype* m)
{
	int i, j;
	int bcindex;
	int tmp, rnd;

	istype info[MAXVERTEX];

	int updown;

	updown = 0;

	/* calculate degrees */
	for (i = 0; i < order; i++) {
		info[i].degree = 0;
		info[i].possible = i;
	}

	for (i = 1; i < order; i++) {
		for (j = 0; j < i; j++) {
			numConfChecks++;
			if (edge(i, j)) {
				info[i].degree++;
				info[j].degree++;
			}
		}
	}
	bcindex = 1;

	//Randomly permute the "info" array before it is sorted according to
	//degree. This means that different seeds can give different orderings of the nodes
	for (i = order - 1; i >= 0; i--) {
		rnd = rand() % (i + 1);
		//Now swap the relevant bists of info
		tmp = info[i].degree; 	info[i].degree = info[rnd].degree; 		info[rnd].degree = tmp;
		tmp = info[i].possible; info[i].possible = info[rnd].possible;	info[rnd].possible = tmp;
	}

	/* sort vertices by degree */
	if (updown == 0)  /* decreasing: THIS IS THE ON */
		qsort(info, order, sizeof(istype), (compfunc)udgcmp);
	else            /* increasing */
		qsort(info, order, sizeof(istype), (compfunc)dgcmp);
	for (i = 0; i < order; i++)
		m->vc[i].vertex = info[i].possible;
	return(bcindex);
}
