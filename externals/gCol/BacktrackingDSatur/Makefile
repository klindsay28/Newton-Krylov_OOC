# Makefile for BacktrackingDSatur

EXEC=BacktrackingDSatur

HEADS=bktdsat.h colorrtns.h graph.h maxclique.h mysys.h

OBJ=bkdmain.o bktdsat.o colorrtns.o graph.o maxclique.o

CPP=g++
OPTS=-std=c++11 -O3 -Wall ${GFLAGS}

all: ${EXEC}

${EXEC}: ${OBJ}
	${CPP} ${OPTS} -o $@ ${OBJ}

%.o: %.cpp ${HEADS}
	${CPP} ${OPTS} -c -o $@ $<

clean:
	rm -f ${OBJ} ${EXEC}
