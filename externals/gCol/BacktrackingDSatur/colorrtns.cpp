#include "mysys.h"
#include "graph.h"
#include "colorrtns.h"
#include <fstream>

/*
A Collection of routines for manipulating vectors of colors, vertices, degree sequences etc. used here and there throughout the system
*/
extern unsigned long long numConfChecks;
extern unsigned long long maxChecks;

void logAndExitPlease(string s) {
	//Writes message s to screen and log file and then exits the program
	ofstream resultsLog("resultsLog.log", ios::app);
	resultsLog << s;
	cout << s;
	resultsLog.close();
	exit(1);
}

/* maximum colors printed per line */
#define MAXPERLINE  20

void getcolorinfo(popmembertype* member)
{
	vertextype i;
	struct vrtxandclr* v;

	/* get color info */
	member->clrdata.numcolors = 1;
	member->clrdata.total = 0;
	v = member->vc;

	for (i = 0; i < order; i++) {
		if (v[i].color > member->clrdata.numcolors)
			member->clrdata.numcolors = v[i].color;
		member->clrdata.total += v[i].color;
	}
	member->clrdata.total += member->clrdata.numcolors * order;
}

/* permute a subrange of a member. Permutes the elements in the range [first..last-1]. All permutations equally likely 			*/
void permute(popmembertype* member, vertextype first, vertextype last)
{
	vertextype i, j;
	struct vrtxandclr k;

	for (i = first; i < last - 1; i++) {
		j = i + ((vertextype)rand() % (last - i));
		k = member->vc[i];
		member->vc[i] = member->vc[j];
		member->vc[j] = k;
	}
}

void trivial_color(popmembertype* m)
{
	vertextype i;
	for (i = 0; i < order; i++)
		m->vc[i].color = i + 1;
}

void verifycolor(popmembertype* m)
{
	char verifyset[MAXVERTEX];
	int clrflagerror;
	int i, j;

	for (i = 0; i < order; i++) verifyset[i] = 0;
	clrflagerror = 0;
	for (i = 0; i < order; i++) {
		verifyset[m->vc[i].vertex] = 1;
		for (j = i + 1; j < order; j++) {
			numConfChecks++;
			if ((edge(m->vc[i].vertex, m->vc[j].vertex)) &&
				(m->vc[i].color == m->vc[j].color))
				clrflagerror++;
		}
	}
	if (clrflagerror > 0) printf("COLORING ERRORS %d\n", clrflagerror);
	else {
		clrflagerror = 0;
		for (i = 0; i < order; i++) if (verifyset[i] == 0) clrflagerror++;
		if (clrflagerror > 0)
			printf("ERROR: %d missing vertices\n", clrflagerror);
		else;
		//printf("Coloring Verified\n");
	}
}

void getacoloring(popmembertype* m, char* name, int* which)
{
	int i, cnt, colnum;
	FILE* fp;
	char fname[100], oneline[100], * s;

	for (i = 0; i < order; i++)
		m->vc[i].vertex = i;

	strncpy(fname, name, 80);
	strncat(fname, ".res", 5);

	if ((fp = fopen(fname, "r")) == NULL) {
		printf("WARNING: getacoloring - cannot open file %s \n", fname);
		printf("\tUSING TRIVIAL COLORING INSTEAD\n");
		trivial_color(m);
	}
	else {
		cnt = 0;
		while (!feof(fp)) {
			s = fgets(oneline, 81, fp);
			if (0 == strncmp(oneline, "CLRS", 4)) {
				cnt++;
				printf("[ %d] %s\n", cnt, oneline);
			}
		}

		if (cnt > 0) {
			printf("There are %d colorings. Which do you want? ", cnt);
			scanf("%d", &colnum);

			/* default take the last */
			if ((colnum > cnt) || (colnum < 1)) colnum = cnt;
			printf("%d\n", colnum);
			*which = colnum;
			rewind(fp);
			cnt = 0;
			while (cnt < colnum) {
				s = fgets(oneline, 81, fp);
				if (0 == strncmp(oneline, "CLRS", 4)) {
					cnt++;
				}
			}
			for (i = 0; i < order; i++) {
				if (0 == fscanf(fp, "%hu", &(m->vc[i].color))) {
					logAndExitPlease("BAD FILE FORMAT\n");
				}
			}
		}
		else {
			printf("NO COLORINGS PRESENT IN FILE\n");
			printf("Using Trivial Color\n");
			trivial_color(m);
		}
		fclose(fp);
	}
}

int degseq[MAXVERTEX + 1];

int decdeg(struct vrtxandclr* a, struct vrtxandclr* b)
/* Comparison routine for sorting by degree downwards */
{
	if (degseq[a->vertex] < degseq[b->vertex]) return(1);
	else if (degseq[a->vertex] == degseq[b->vertex]) return(0);
	else return(-1);
}

int incdeg(struct vrtxandclr* a, struct vrtxandclr* b)
/* Comparison routine for sorting by degree upwards */
{
	if (degseq[a->vertex] > degseq[b->vertex]) return(1);
	else if (degseq[a->vertex] == degseq[b->vertex]) return(0);
	else return(-1);
}

void computedeg()
/* Compute degree sequence. */
{
	int i, j;
	adjacencytype* x;
	for (i = 0; i < order; i++) degseq[i] = 0;
	for (i = 1; i < order; i++) {
		numConfChecks++;
		initnbr(x, i);
		for (j = 0; j < i; j++) {
			numConfChecks++;
			if (isnbr(x, j)) {
				degseq[i]++;
				degseq[j]++;
			}
		}
	}
}

void fileres(char* name, popmembertype* m, char* prgm)
{
	//Print the solution to a file
	int i;
	colortype c[MAXVERTEX];

	for (i = 0; i < order; i++) {
		c[m->vc[i].vertex] = m->vc[i].color;
	}

	ofstream solStrm;
	solStrm.open("solution.txt");
	solStrm << order << "\n";
	for (i = 0; i < order; i++) solStrm << c[i] - 1 << endl;
	solStrm.close();
}
