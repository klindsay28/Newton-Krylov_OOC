# Makefile for AntCol

EXEC=AntCol

HEADS=Graph.h inputGraph.h makesolution.h manipulateArrays.h tabu.h

OBJ=Graph.o inputGraph.o main.o makesolution.o manipulateArrays.o tabu.o

CPP=g++
OPTS=-std=c++11 -O3 -Wall ${GFLAGS}

all: ${EXEC}

${EXEC}: ${OBJ}
	${CPP} ${OPTS} -o $@ ${OBJ}

%.o: %.cpp ${HEADS}
	${CPP} ${OPTS} -c -o $@ $<

clean:
	rm -f ${OBJ} ${EXEC}
