#pragma once

#define NFA_SIZE 32

typedef struct NFA {
	int accept[NFA_SIZE];
	int transitions[NFA_SIZE][256][NFA_SIZE];

} nfa;

nfa* makeNFA();