﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifndef __CUDACC__
#define __CUDACC__
#include <device_functions.h>
#endif

#include <stdio.h>
#include<iostream>

#include "dfa.h"
#include "nfa.h"
#include "nfa_loader.h"
#include "book_loader.h"
#include "infant.h"

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
__global__ void runNFAGPU(nfa* n, char* s, int length) {

	int* active_states = (int*)malloc(NFA_SIZE * sizeof(int));
	memset(active_states, 0, NFA_SIZE * sizeof(int));
	active_states[0] = 1; //start state active

	for (int i = 0; i < length; i++) {

		int* new_states = (int*)malloc(NFA_SIZE * sizeof(int));
		memset(new_states, 0, NFA_SIZE * sizeof(int));
		printf("GPU: reading char %c.active states:", s[i]);
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
			printf("GPU state %d active", i);
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

void launchNFA(nfa* n, char* str, int len, int blocks, int threadsPerBlock, float* memoryTime, float* computationTime)
{
	char* dev_str = nullptr;
	nfa* dev_nfa = nullptr;

	

	cudaEvent_t memoryStart, memoryStop; //track memory
	cudaEventCreate(&memoryStart);
	cudaEventCreate(&memoryStop);


	cudaEvent_t computeStart, computeStop; //track compute
	cudaEventCreate(&computeStart);
	cudaEventCreate(&computeStop);

	cudaMalloc((void**)& dev_nfa, 1 * sizeof(nfa)); //allocate device memory
	cudaMalloc((void**)& dev_str, len * sizeof(char));
	cudaEventRecord(memoryStart); //record start of memory 

	cudaMemcpy(dev_nfa, n, 1 * sizeof(nfa), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_str, str, len * sizeof(char), cudaMemcpyHostToDevice);
	
	cudaEventRecord(memoryStop); //record end of memory stuff, use event synch to get correct time
	cudaEventSynchronize(memoryStop);
	

	//We use event sync instead of device sync because eveny stync will freeze the CPU thread just like device
	// But with the added benefit freezing until the event recording, which is right after the kernel finishes


	cudaEventRecord(computeStart); //same procedure for running NFA
	runNFAGPU << <blocks, threadsPerBlock >> > (dev_nfa, dev_str, len);
	cudaEventRecord(computeStop);
	cudaEventSynchronize(computeStop);
	
	
	cudaEventElapsedTime(memoryTime, memoryStart, memoryStop); //see results
	cudaEventElapsedTime(computationTime, computeStart, computeStop);
	
	printf("Memory Took: %f ms\n", *memoryTime);
	printf("Computation Took: %f ms\n", *computationTime);
	//clean up
	cudaFree(dev_str);
	cudaFree(dev_nfa);
}

__global__ void infantAlgorithm(INFANT* nfa, char* book, int bookLength,  char* active, char* future) {
	//active and future are both assumed to be nfa state sized
	

	__shared__ int i;
	i = 0;

	//for(int i = 0; i < bookLength; i++){
	while (i < bookLength){
	//start in a block given by index and go by block width
		char c = book[i];
		printf("i is %d according to thread %d and c is %c, bdx is %d \n", i, threadIdx.x,c, blockDim.x);
		//TODO max states
		for (int j = threadIdx.x; j < nfa->maxTransitions[c-FIRST_CHAR]; j += blockDim.x) {
			
			//So here we have 2 state IDs stored together and they're each 16 bits and stored in one 32 bit int
			// the lower 16 are the start and the upper 16 are the end
			// So we get a pointer to that int and then use short pointers to the top and bottom to get the states
			
			
			short* startState;
			short* endState;
			
			int* transition = &(nfa->transitions[c-FIRST_CHAR][j]);
			//printf("thread %d is looking at transition %.8x\n", threadIdx.x, *transition);
			
				
				startState = ((short*)transition)+1; //the delights of endianess make you do this at least on my AMD machine
				endState = ((short*)transition);

				int start = (int)(*startState);
				int end = (int)(*endState);
				//printf("before checking state current is { %d, %d} and future is {%d, %d}\n",(int) active[0], (int)active[1], (int)future[0], (int)future[1]);
				if (active[start] != 0) { //if current state in transition is active then future is active
					future[end] = 1;
					
					//printf("in state %d with char %c moving to %d via transition %d in thread %d and i is %d\n", start, c, end, j, threadIdx.x ,i);
					
				}
				//printf("after checking state current is { %d, %d} and future is {%d, %d}\n", (int)active[0], (int)active[1], (int)future[0], (int)future[1]);
			
		}
		
		__syncthreads();//copy future to current
		for (int j = threadIdx.x; j < MAX_STATES; j+= blockDim.x){
			active[j] = future[j];
			future[j] = 0;
		}
	
		
		__syncthreads(); //wait till all threads are done, then incremenet i, and syc again

		if (threadIdx.x == 0) {
			i++;

		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		for (int i = 0; i < 10; i++) {
			printf("in state %d, with setting %d\n", i, active[i]);
		}
	}



}

void runInfant(INFANT* nfa, char* book, int bookLength, float* memoryTime, float* computationTime) {
	int firsts[NFA_CHARS];
	for (int i = 0; i < NFA_CHARS; i++) {
		firsts[i] = nfa->transitions[i][0]; //copy first transitions to a list of first transitions
	}


	int blocks = 1;
	int threadsPerBlock = 1;

	char* dev_book = nullptr;
	INFANT* dev_nfa = nullptr;
	

	char current_states[MAX_STATES] = { 0 };
	current_states[0] = 1;

	char* dev_current_states = nullptr; // allocate state array
	cudaMalloc((void**)& dev_current_states, MAX_STATES * sizeof(char));
	cudaMemcpy(dev_current_states, current_states, MAX_STATES * sizeof(char), cudaMemcpyHostToDevice);

	char future_states[MAX_STATES] = { 0 };
	

	char* dev_future_states = nullptr; // allocate state array
	cudaMalloc((void**)& dev_future_states, MAX_STATES * sizeof(char));
	cudaMemcpy(dev_future_states,future_states, MAX_STATES * sizeof(char), cudaMemcpyHostToDevice);


	cudaEvent_t memoryStart, memoryStop; //track memory
	cudaEventCreate(&memoryStart);
	cudaEventCreate(&memoryStop);


	cudaEvent_t computeStart, computeStop; //track compute
	cudaEventCreate(&computeStart);
	cudaEventCreate(&computeStop);

	cudaMalloc((void**)& dev_nfa, 1 * sizeof(INFANT)); //allocate device memory
	cudaMalloc((void**)& dev_book, bookLength * sizeof(char));
	
	cudaEventRecord(memoryStart); //record start of memory 

	cudaMemcpy(dev_nfa, nfa, 1 * sizeof(INFANT), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_book, book, bookLength * sizeof(char), cudaMemcpyHostToDevice);

	cudaEventRecord(memoryStop); //record end of memory stuff, use event synch to get correct time
	cudaEventSynchronize(memoryStop);


	//We use event sync instead of device sync because eveny sync will freeze the CPU thread just like device
	// But with the added benefit freezing until the event recording, which is right after the kernel finishes


	cudaEventRecord(computeStart); //same procedure for running NFA
	//runNFAGPU << <blocks, threadsPerBlock >> > (dev_nfa, dev_str, len);
	infantAlgorithm << <blocks, threadsPerBlock >> > (dev_nfa, dev_book, bookLength, dev_current_states, dev_future_states);
	cudaEventRecord(computeStop);
	cudaEventSynchronize(computeStop);


	cudaEventElapsedTime(memoryTime, memoryStart, memoryStop); //see results
	cudaEventElapsedTime(computationTime, computeStart, computeStop);

	printf("Memory Took: %f ms\n", *memoryTime);
	printf("Computation Took: %f ms\n", *computationTime);
	//clean up
	cudaFree(dev_book);
	cudaFree(dev_nfa);
	cudaFree(dev_current_states);
	cudaFree(dev_future_states);


}

int main()
{

	std::string s = loadBook("D:/CUDFA/CUDFA/x64/Debug/romeo_and_juliet.txt");
	std::string* s2 = &s;
	int char_count;
	char* book = processBook(s2, &char_count);

	iNFAnt* nfa = getiNFAnt();
	nfa->transitions['a' - FIRST_CHAR][0] = 0x00000000; //loop from 0 to 0 when you see a
	nfa->transitions['a' - FIRST_CHAR][1] = 0x00010000; // when you see a at state 1 go to 0

	nfa->transitions['b' - FIRST_CHAR][0] = 0x00010001; // when you see b at state 1 loop
	nfa->transitions['b' - FIRST_CHAR][1] = 0x00000001; // when you see b at state 0 go to state 1
	nfa->transitions['b' - FIRST_CHAR][2] = 0x00020001; // when you see b at state 2 go state 1

	nfa->transitions['c' - FIRST_CHAR][0] = 0x00020002; // when you see b at state 2 loop
	nfa->transitions['c' - FIRST_CHAR][1] = 0x00000002; // when you see b at state 0 go to state 2
	nfa->transitions['c' - FIRST_CHAR][2] = 0x00010002;  // c at state 1 means go to state 2


	nfa->maxTransitions['a' - FIRST_CHAR] = 2;
	nfa->maxTransitions['b' - FIRST_CHAR] = 3;
	nfa->maxTransitions['c' - FIRST_CHAR] = 3;

	char* str = "abaaaabc";

	iNFAnt* nfa2 = getiNFAnt();

	//romeos goes from 0 to 5
	nfa2->transitions['r' - FIRST_CHAR][0] = 0x00000001; // r means 0 to 1
	nfa2->transitions['o' - FIRST_CHAR][0] = 0x00010002;
	nfa2->transitions['m' - FIRST_CHAR][0] = 0x00020003;
	nfa2->transitions['e' - FIRST_CHAR][0] = 0x00030004;
	nfa2->transitions['o' - FIRST_CHAR][1] = 0x00040005;

	float memoryTime;
	float computationTime;
	char* st = "romeo";
	runInfant(nfa2, st, strlen(st), &memoryTime, &computationTime);




	getchar();
	return 0;

	/**
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
	//dfa* d = makeDFA();
	//d->accept[1] = 1;
	//d->transitions[0]['a'] = 1;// chars are numbers, very hacky
	//d->transitions[0]['b'] = 0;
	//d->transitions[1]['a'] = 1;
	//d->transitions[1]['b'] = 0; //*a
	char* str = "baab";
	//runDFA(d, str, strlen(str));
	//
	//
	//dfa* dev_dfa = nullptr;
	//char* dev_str = nullptr;
	//
	//cudaMalloc((void**)& dev_dfa, 1 * sizeof(dfa));
	//cudaMalloc((void**)& dev_str, strlen(str) * sizeof(char));
	//
	//cudaMemcpy(dev_dfa, d, 1 * sizeof(dfa), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_str, str, strlen(str) * sizeof(char), cudaMemcpyHostToDevice);
	//DFAGPU << <1, 1 >> > (dev_dfa, dev_str, strlen(str));
	//cudaDeviceSynchronize();


	
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
	float memoryTime = 1;
	float computeTime;
	launchNFA(n, str, strlen(str), 1, 1, &memoryTime, &computeTime); */

	
	
}






