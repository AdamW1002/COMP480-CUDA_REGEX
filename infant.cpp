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

	return infant;

}
