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
	int selfLoops[MAX_STATES]; //ancillary array of self loop states

} iNFAnt;

iNFAnt* getiNFAnt();