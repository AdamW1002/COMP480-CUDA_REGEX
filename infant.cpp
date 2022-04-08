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

	memset(infant->transitions, -1, NFA_CHARS * MAX_STATES * sizeof(int)); //0 out the nfa

	return infant;

}
