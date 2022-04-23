#pragma once
//This is an implementation of an iNFAnt style NFA
// https://dl.acm.org/doi/pdf/10.1145/1880153.1880157?casa_token=ykqhZMnv1-gAAAAA:Ev99A-poJgSTBatXHqGJsdE77GEhRfu8-nJtVxs60eX3udOR09q6wGcdtH21-vBsgDof4028fH7t
// The paper for your viewing pleasure

#define NFA_CHARS 95 //There are only 95 chars in the NFA's char set because we limit ourselves to reasonable chars 
#define MAX_STATES 1 << 16 //this is just how many states there are per the paper
#define FIRST_CHAR 32 //first char read by NFA
typedef struct INFANT {

	int transitions[NFA_CHARS][MAX_STATES]; //look up char first then choose from 65536 states per infant paper also each int is a pair of 2 16 bit numbers
	short  acceptStates[MAX_STATES];
	char selfLoops[MAX_STATES]; //ancillary array of self loop states
	int maxTransitions[NFA_CHARS]; //record index of highest transition
	short maxState; //highest state reached by nfa

} iNFAnt;

iNFAnt* getiNFAnt(); //get an infant pre aloocated
void addTransition(iNFAnt* nfa,char c, short start, short end); //add a transition and avoid a mess
void addString(iNFAnt* nfa, char* str, int start); //add a string to our NFA
void addEpsilon(iNFAnt* nfa, int start,int  end); //add epsilon transition from start to end
int addEpsilonString(iNFAnt* nfa, int start, int  count);
void addGroupOfMany(iNFAnt* nfa, int start, int min, int max, int* end);