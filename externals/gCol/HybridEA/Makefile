# Makefile for HybridEA

EXEC=HybridEA

HEADS=diversity.h Graph.h hungarian.h inputGraph.h Kempe.h makesolution.h manipulateArrays.h tabu.h xover.h

OBJ=diversity.o Graph.o hungarian.o inputGraph.o Kempe.o main.o makesolution.o manipulateArrays.o tabu.o xover.o

CPP=g++
OPTS=-std=c++11 -O3 -Wall ${GFLAGS}

all: ${EXEC}

${EXEC}: ${OBJ}
	${CPP} ${OPTS} -o $@ ${OBJ}

%.o: %.cpp ${HEADS}
	${CPP} ${OPTS} -c -o $@ $<

clean:
	rm -f ${OBJ} ${EXEC}
