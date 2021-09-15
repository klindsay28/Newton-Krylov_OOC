# Makefile for SimpleGreedy

EXEC=Constructive

HEADS=

OBJ=main.o

CPP=g++
OPTS=-std=c++11 -O3 -Wall ${GFLAGS}

all: ${EXEC}

${EXEC}: ${OBJ}
	${CPP} ${OPTS} -o $@ ${OBJ}

%.o: %.cpp ${HEADS}
	${CPP} ${OPTS} -c -o $@ $<

clean:
	rm -f ${OBJ} ${EXEC}
