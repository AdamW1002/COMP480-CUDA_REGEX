#include "nfa.h"
#include "nfa.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

nfa* makeNFA() {

	nfa* n = (nfa*)malloc(1 * sizeof(nfa));
	if (n == NULL) {
		printf("NFA allocation failed\n");
	}
	else {
		memset(n, -1, 1 * sizeof(nfa)); //make transitions -1 and accept states all false by default
		memset(n->accept, 0, NFA_SIZE * sizeof(int));
	}
	return n;
}

