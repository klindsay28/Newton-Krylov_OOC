# Makefile for HillClimber

EXEC=HillClimber

HEADS=algfns.h display.h hillClimber.h initialsolutiongenerator.h

OBJ=algfns.o display.o hillClimber.o initialsolutiongenerator.o

CPP=g++
OPTS=-std=c++11 -O3 -Wall ${GFLAGS}

all: ${EXEC}

${EXEC}: ${OBJ}
	${CPP} ${OPTS} -o $@ ${OBJ}

%.o: %.cpp ${HEADS}
	${CPP} ${OPTS} -c -o $@ $<

clean:
	rm -f ${OBJ} ${EXEC}
