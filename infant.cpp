#include "infant.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

iNFAnt* getiNFAnt()
{
	iNFAnt* infant = (iNFAnt*)malloc(1 * sizeof(iNFAnt));
	if (infant == NULL) {
		printf("Failed to allocate infant nfa\n");
		return NULL;
	}

	memset(infant, 0, 1 * sizeof(infant)); //0 out the nfa
	memset(infant->maxTransitions, 0, NFA_CHARS * sizeof(int));
	//memset(infant->transitions, 0, NFA_CHARS *  MAX_STATES * sizeof(int));
	memset(infant->selfLoops, 0, MAX_STATES * sizeof(char));
	infant->maxState = 0;//we assume theres alwas a start state
	return infant;

}

void addTransition(iNFAnt* nfa, char c, short start, short end)
{
	short arr[2] = { end, start }; //array and pointer trick to form int from two shorts
	int* i = (int*)arr;


	nfa->transitions[c - FIRST_CHAR][nfa->maxTransitions[c - FIRST_CHAR]] = *i; //add transition and then denote that maxtransitions increases
	nfa->maxTransitions[c - FIRST_CHAR]++;
	if (end > nfa->maxState || start > nfa->maxState) {
	
		if (end > start){
			nfa->maxState = end;

		}
		else {
			nfa->maxState = start;
		}

	}
}

