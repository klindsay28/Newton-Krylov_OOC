#ifndef GRAPHDEFS
#define GRAPHDEFS

#define MAXVERTEX  8000
#define SHIFT 3
#define MASK 7
#define ROWSIZE ((MAXVERTEX >> SHIFT) +1 )
#define GRAPHSIZE  (MAXVERTEX * ROWSIZE)

/* Definitions useful for checking and setting edges */

/* set and clear are asymmetric - use setedge(i,j) setedge(j,i) etc. */
#define setedge(i,j)  graph[((i)*ROWSIZE) + ((j) >> SHIFT)] |= (1 << ((j) & MASK))
#define clearedge(i,j) graph[((i)*ROWSIZE) + ((j) >> SHIFT)] &= ~(1 << ((j) & MASK))
#define edge(i,j)    (graph[((i)*ROWSIZE)+((j) >> SHIFT)] & (1 << ((j) & MASK)) )

/* for loops involving potential neighbors */
#define initnbr(x,i)  (x) = graph + ((i)*ROWSIZE)
#define isnbr(x,i)  (((x)[(i) >> SHIFT]) & (1 << ((i) & MASK)))

typedef int vertextype;
typedef unsigned char adjacencytype;
extern adjacencytype graph[GRAPHSIZE];
extern vertextype order;

/* CHEAT INFORMATION FROM GRAPH */
extern int partset[MAXVERTEX];
extern int partitionflag;
extern int partitionnumber;
extern void getgraph(char a[]);

#endif
