# This makefile builds all the graph coloring executables

GLOBALFLAGS := "-Wno-sign-compare -Wno-unused-but-set-variable"
GLOBALFLAGS := "-Wno-sign-compare"

subdirs := AntCol BacktrackingDSatur Constructive HillClimber HybridEA PartialColAndTabuCol

.PHONY: all ${subdirs} clean

all: ${subdirs}

${subdirs}:
	${MAKE} -C $@ GFLAGS=${GLOBALFLAGS}

clean:
	for dir in ${subdirs}; do \
          $(MAKE) -C $$dir clean; \
        done
