#pragma once
#define DFA_SIZE 32
typedef struct DFA {

	
	int accept[DFA_SIZE];

	int transitions[DFA_SIZE][256];


} dfa;

dfa* makeDFA();