# Makefile for PartialColAndTabuCol

EXEC=PartialColAndTabuCol

HEADS=Graph.h initializeColoring.h inputGraph.h manipulateArrays.h reactcol.h tabu.h

OBJ=Graph.o initializeColoring.o inputGraph.o main.o manipulateArrays.o reactcol.o tabu.o

CPP=g++
OPTS=-std=c++11 -O3 -Wall ${GFLAGS}

all: ${EXEC}

${EXEC}: ${OBJ}
	${CPP} ${OPTS} -o $@ ${OBJ}

%.o: %.cpp ${HEADS}
	${CPP} ${OPTS} -c -o $@ $<

clean:
	rm -f ${OBJ} ${EXEC}
