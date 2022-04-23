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

		if (end > start) {
			nfa->maxState = end;

		}
		else {
			nfa->maxState = start;
		}

	}
}

void addString(iNFAnt* nfa, char* str, int start){

	
	for (int i = 0; i < strlen(str); i++) { //for each char connect state to next
		int state = start + i;
		addTransition(nfa, str[i], state, state + 1);
		
	}
	
}

void addEpsilon(iNFAnt* nfa, int start, int end){

	for (char c = FIRST_CHAR; c < FIRST_CHAR + NFA_CHARS; c++) {
		addTransition(nfa, c, start, end);
	}

}

int addEpsilonString(iNFAnt* nfa, int start, int count){
	int state;
	for (int i = 0; i < count; i++) {
		addEpsilon(nfa, start + i, start + i + 1);
		state = start + i + 1;
		if (state > nfa->maxState || start > nfa->maxState) {

			if (state > start) {
				nfa->maxState = state;

			}
			else {
				nfa->maxState = start;
			}

		}
	}

	return state;
}

void addGroupOfMany(iNFAnt* nfa, int start, int min, int max, int* end){
	int initial = nfa->maxState + 1; //first unused state
	addEpsilon(nfa, start, initial);
	for (int i = min; i < max; i++) { //go thru range
		int* endPoints = (int*) malloc((max-min) * sizeof(int)); //store where we end
		for (int j = 1; j < max - min; j++) {
			
			for (int k = 0; k < j; k++) {
			
			}

		}
	
	}

}

