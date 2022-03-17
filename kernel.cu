﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<iostream>

#include "dfa.h"
#include "nfa.h"


__global__ void addi(int* x, int* y, int* z) {

	int i = threadIdx.x;
	z[i] = x[i] + y[i];
	printf("hello gpu thread %d\n", i);

}
void runDFA(dfa* d, char* s, int length) {
	int state = 0;
	for (int i = 0; i < length; i++) {
		printf("current state is %d, reading char %c ", state, s[i]);
		state = d->transitions[state][s[i]];
		printf("moving to %d\n", state);
	}
	if (d->accept[state]) {

		printf("accepting\n");
	}
	else {
		printf("rejecting string %s \n", s);
	}

}


__global__ void DFAGPU(dfa* d, char* s, int length) {
	int state = 0;
	for (int i = 0; i < length; i++) {
		printf("current state is %d, reading char %c ", state, s[i]);
		state = d->transitions[state][s[i]];
		printf("moving to %d\n", state);
	}
	printf("final state %d \n", state);
	if (d->accept[state] != 0) {

		printf("accepting on gpu\n");
	}
	else {
		printf("rejecting string %s on gpu \n", s);
	}

}


void runNFA(nfa* n, char* s, int length) {

	int* active_states = (int*)malloc(NFA_SIZE * sizeof(int));
	memset(active_states, 0, NFA_SIZE * sizeof(int));
	active_states[0] = 1; //start state active

	for (int i = 0; i < length; i++) {

		int* new_states = (int*)malloc(NFA_SIZE * sizeof(int));
		memset(new_states, 0, NFA_SIZE * sizeof(int));
		printf("reading char %c.active states:", s[i]);
		for (int j = 0; j < NFA_SIZE; j++) { //go thru active states
			if (active_states[j] == 1) {
				printf(" %d", j);


				for (int k = 0; k < NFA_SIZE; k++) { //go thru possible transitions
					
					if (n->transitions[j][s[i]][k] == 1) {
						
						new_states[k] = 1;
					}

				}
			}

			

		}
		printf("\n");
		free(active_states);
		active_states = new_states;

	}

	int accepted = 0;
	for (int i = 0; i < NFA_SIZE; i++) {
		if (active_states[i] == 1) {
			printf("state %d active", i);
			if (n->accept[i]) {
				printf(" and accepting");
				accepted = 1;
			}
			printf("\n");
		}
	}
	
	if (accepted == 0) {
		printf("no active states, rejecting %s \n", s);
	}

}
//USE CUDA TIMERS
//CudaEvent
int main()
{

	int x[3] = { 1,2,3 };
	int y[3] = { 4,5,6 };
	int z[3] = { 0 };

	int* dev_z = nullptr;
	int* dev_x = nullptr;
	int* dev_y = nullptr;

	cudaMalloc((void**)& dev_x, 3 * sizeof(int));
	cudaMalloc((void**)& dev_y, 3 * sizeof(int));
	cudaMalloc((void**)& dev_z, 3 * sizeof(int));


	printf("%d\n", cudaGetLastError());
	cudaMemcpy(dev_x, x, 3 * sizeof(int), cudaMemcpyHostToDevice);
	printf("%d\n", cudaGetLastError());
	cudaMemcpy(dev_y, y, 3 * sizeof(int), cudaMemcpyHostToDevice);

	printf("%d\n", cudaGetLastError());
	addi << <1, 3 >> > (dev_x, dev_y, dev_z);
	printf("%d\n", cudaGetLastError());
	cudaDeviceSynchronize();
	cudaMemcpy(&z, dev_z, 3 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", cudaGetLastError());
	for (int i = 0; i < 3; i++) {
		printf("%d\n", z[i]);
	}


	//dfa* d = (dfa*)malloc(1 * sizeof(dfa)); //alocate dfa and make sure its all -1
	//memset(d, -1, 1 * sizeof(dfa)); //make accept states not true
	//memset(d->accept, 0, DFA_SIZE * sizeof(int));
	dfa* d = makeDFA();
	d->accept[1] = 1;
	d->transitions[0]['a'] = 1;// chars are numbers, very hacky
	d->transitions[0]['b'] = 0;
	d->transitions[1]['a'] = 1;
	d->transitions[1]['b'] = 0; //*a
	char* str = "baab";
	runDFA(d, str, strlen(str));


	dfa* dev_dfa = nullptr;
	char* dev_str = nullptr;

	cudaMalloc((void**)& dev_dfa, 1 * sizeof(dfa));
	cudaMalloc((void**)& dev_str, strlen(str) * sizeof(char));

	cudaMemcpy(dev_dfa, d, 1 * sizeof(dfa), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_str, str, strlen(str) * sizeof(char), cudaMemcpyHostToDevice);
	DFAGPU << <1, 1 >> > (dev_dfa, dev_str, strlen(str));
	cudaDeviceSynchronize();


	
	nfa* n = makeNFA();
	n->accept[1] = 1;
	n->accept[2] = 1;
	n->transitions[0]['a'][1] = 1; //from state 0 when it sees a go to state 1 or 2
	n->transitions[0]['a'][2] = 1;

	n->transitions[1]['a'][1] = 1; //when in accept state, stay there 
	n->transitions[1]['a'][2] = 1;
	n->transitions[2]['a'][1] = 1;
	n->transitions[2]['a'][2] = 1;

	n->transitions[0]['b'][0] = 1;//when theres a b always go to a reject state
	n->transitions[1]['b'][0] = 1;
	n->transitions[2]['b'][0] = 1;

	runNFA(n, str, strlen(str));

	getchar();
	return 0;
}






