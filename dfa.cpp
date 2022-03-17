#include "dfa.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

dfa* makeDFA() {
	dfa* d =  (dfa*)malloc(1 * sizeof(dfa));
	if (d == NULL) {
		printf("DFA allocation failed\n");
	}
	else {
		memset(d, -1, 1 * sizeof(dfa)); //make accept states not true but everything else -1 to avoid transitions to 0
		memset(d->accept, 0, DFA_SIZE * sizeof(int));
	}
	return d;
}
