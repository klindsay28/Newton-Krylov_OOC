#ifndef MYSYS
#define MYSYS 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>

using namespace std;

/* time information */
extern long seconds, microsecs;
extern struct rusage tmp;

#include <signal.h>

/* the following required to make qsort calling args silent */
typedef int (*compfunc)(const void*, const void*);

#endif
