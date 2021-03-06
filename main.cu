
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifndef __CUDACC__
#define __CUDACC__
#include <device_functions.h>
#endif

#include <stdio.h>
#include<iostream>
#include <map>

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

__global__ void infantAlgorithm(INFANT* nfa, char* book, int bookLength, char* active, char* future, int* acceptCounts) {
	//active and future are both assumed to be nfa state sized


	__shared__ int i;
	i = 0;
	__shared__ char selfLoop[256];
	if (nfa->maxState <= 256)
	{
		for (int i = threadIdx.x; i <= nfa->maxState; i += blockDim.x) {

			selfLoop[i] = book[i];
		}
	}


	//for(int i = 0; i < bookLength; i++){
	while (i < bookLength) {
		//start in a block given by index and go by block width
		char c = book[i];
		//printf("i is %d according to thread %d and c is %c, bdx is %d \n", i, threadIdx.x,c, blockDim.x);
		//TODO max states
		for (int j = threadIdx.x; j < nfa->maxTransitions[c - FIRST_CHAR]; j += blockDim.x) {

			//So here we have 2 state IDs stored together and they're each 16 bits and stored in one 32 bit int
			// the lower 16 are the start and the upper 16 are the end
			// So we get a pointer to that int and then use short pointers to the top and bottom to get the states


			short* startState;
			short* endState;
			//load as int and instead shift+mask
			/**int* transition = &(nfa->transitions[c-FIRST_CHAR][j]);
			//printf("thread %d is looking at transition %.8x\n", threadIdx.x, *transition);


				startState = ((short*)transition)+1; //the delights of endianess make you do this at least on my AMD machine
				endState = ((short*)transition);

				int start = (int)(*startState);
				int end = (int)(*endState);**/
			int transition = (nfa->transitions[c - FIRST_CHAR][j]); //use bitshifts to decompose intger into high and low bits
			int start = (transition & 0xFFFF0000) >> 16;
			int end = (transition & 0x0000FFFF);
			//printf("before checking state current is { %d, %d} and future is {%d, %d}\n",(int) active[0], (int)active[1], (int)future[0], (int)future[1]);
			if (active[start] != 0) { //if current state in transition is active then future is active
				future[end] = 1;


				//printf("in state %d with char %c moving to %d via transition %d in thread %d and i is %d\n", start, c, end, j, threadIdx.x ,i);

			}
			//printf("after checking state current is { %d, %d} and future is {%d, %d}\n", (int)active[0], (int)active[1], (int)future[0], (int)future[1]);

		}


		//make sure future is totally done
		__syncthreads();//copy future to current
		for (int j = threadIdx.x; j <= nfa->maxState; j += blockDim.x) {
			active[j] = future[j];
			//if (nfa->maxState <= 256) {
			//	active[j] = active[j] | selfLoop[j];
			//}
			//else {
			active[j] = active[j] | nfa->selfLoops[j]; //if in self loop continue to run
		//}
		//if (nfa->acceptStates[j] == 1 && active[j] != 0) { //if going to be in accpet state count it
		//	acceptCounts[j] = acceptCounts[j] + 1;
		//	
		//}
			acceptCounts[j] += nfa->acceptStates[j] == 1 && active[j] != 0;
			future[j] = 0;
		}



		//no consistent view between thread blocks
		if (threadIdx.x == 0) {
			i++;

		}
		//make sure threads are on same iteration
		__syncthreads();
	}

	//if (threadIdx.x == 0) {
	//	for (int i = 0; i <= nfa->maxState; i++) {
	//		printf("in state %d, with setting %d\n", i, active[i]);
	//	}
	//	for (int i = 0; i <= nfa->maxState; i++) {
	//		printf("state %d active count is %d\n", i, acceptCounts[i]);
	//	}
	//
	//}



}

void runInfant(INFANT* nfa, char* book, int bookLength, float* memoryTime, float* computationTime, int blocks, int threadsPerBlock) {
	int firsts[NFA_CHARS];
	for (int i = 0; i < NFA_CHARS; i++) {
		firsts[i] = nfa->transitions[i][0]; //copy first transitions to a list of first transitions
	}




	char* dev_book = nullptr;
	INFANT* dev_nfa = nullptr;


	int* dev_counts = nullptr; //allocate active counter
	cudaMalloc((void**)& dev_counts, nfa->maxState * sizeof(int));
	cudaMemset(dev_counts, 0, nfa->maxState * sizeof(int));
	//counts for analysis 
	int* counts = (int*)malloc(nfa->maxState * sizeof(int));

	char current_states[MAX_STATES] = { 0 };
	current_states[0] = 1;

	char* dev_current_states = nullptr; // allocate state array
	cudaMalloc((void**)& dev_current_states, MAX_STATES * sizeof(char));
	cudaMemcpy(dev_current_states, current_states, MAX_STATES * sizeof(char), cudaMemcpyHostToDevice);

	char future_states[MAX_STATES] = { 0 };


	char* dev_future_states = nullptr; // allocate state array
	cudaMalloc((void**)& dev_future_states, MAX_STATES * sizeof(char));
	cudaMemcpy(dev_future_states, future_states, MAX_STATES * sizeof(char), cudaMemcpyHostToDevice);


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
	infantAlgorithm << <blocks, threadsPerBlock >> > (dev_nfa, dev_book, bookLength, dev_current_states, dev_future_states, dev_counts);
	cudaEventRecord(computeStop);
	cudaEventSynchronize(computeStop);


	cudaEventElapsedTime(memoryTime, memoryStart, memoryStop); //see results
	cudaEventElapsedTime(computationTime, computeStart, computeStop);

	//printf("Memory Took: %f ms\n", *memoryTime);
	//printf("Computation Took: %f ms\n", *computationTime);
	//
	//cudaMemcpy(counts, dev_counts, nfa->maxState * sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < nfa->maxState; i++) {
	//	printf("state %d count is %d\n", i, counts[i]);
	//}

	//clean up
	cudaFree(dev_book);
	cudaFree(dev_nfa);
	cudaFree(dev_current_states);
	cudaFree(dev_future_states);


}



std::string runExperiment(const char* book_title, iNFAnt* automaton) {

	std::string s = loadBook(book_title);
	std::string* s2 = &s;
	int char_count;
	char* book = processBook(s2, &char_count);

	float memoryTime;
	float computationTime;
	char* st = "romeo and juliet died";
	//printf("book is %d long\n", char_count);
	//runInfant(nfa2, st, strlen(st), &memoryTime, &computationTime);
	
	std::map<int, float> threads_to_time;
	int trials = 5;
	for (int j = 0; j < trials; j++) {
		for (int i = 1; i <= 32; i++) {
			runInfant(automaton, book, char_count, &memoryTime, &computationTime, 1, i);
			if (threads_to_time.find(i) != threads_to_time.end()) {
				threads_to_time[i] = 0.0;
			}
			threads_to_time[i] += computationTime / trials;
			
		}
	}
	std::string experiment_data = "{";
	for (auto it = threads_to_time.begin(); it != threads_to_time.end(); ++it) {
		experiment_data += +", " + std::to_string(it->first) + " : " + std::to_string(it->second);
	}
	experiment_data.replace(1, 2, "");
	return experiment_data + "}";



}

int main()
{

	iNFAnt* experimentalNFA = getiNFAnt();
	addEpsilon(experimentalNFA, 0, 0);//always loop beginning 

	addString(experimentalNFA, "romeo", 0); //Look for romeo in all books, helps as debuggin sanity check
	addTransition(experimentalNFA, 'R', 0, 1); //capital

	int romeoAccept = experimentalNFA->maxState;
	experimentalNFA->acceptStates[romeoAccept] = 1; //accept romeo
	//experimentalNFA->acceptStates[romeoAccept + 1] = 1; //count chars (6)
	std::cout << "romeo accept state is " << romeoAccept << std::endl;
	addTransition(experimentalNFA, 'O', 0, 7); //Now try OF THE
	addTransition(experimentalNFA, 'o', 0, 7);
	addString(experimentalNFA, "f the", 7); //search for the string "of the"
	int ofTheAccept = experimentalNFA->maxState;
	experimentalNFA->acceptStates[ofTheAccept] = 1; //12
	std::cout << "of the accept state is " << ofTheAccept << std::endl;
	//13 for next state

	addTransition(experimentalNFA, 'T', 0, 14);
	addTransition(experimentalNFA, 't', 0, 14);
	addTransition(experimentalNFA, 'h', 14, 15);
	addTransition(experimentalNFA, 'e', 15, 16);
	addTransition(experimentalNFA, 'r', 16, 17);
	addTransition(experimentalNFA, 'e', 17, 18);
	int thereAccept = experimentalNFA->maxState;
	//experimentalNFA->acceptStates[13] = 1;
	//experimentalNFA->acceptStates[14] = 1;
	//experimentalNFA->acceptStates[15] = 1;
	//experimentalNFA->acceptStates[16] = 1;
	//experimentalNFA->acceptStates[17] = 1;
	experimentalNFA->acceptStates[thereAccept] = 1;
	//addEpsilon(experimentalNFA, 0, thereAccept);
	std::cout << "there accept state is" << thereAccept << std::endl;


	addTransition(experimentalNFA, 'i', 16, 20);
	addTransition(experimentalNFA, 'r', 20, 21);
	experimentalNFA->acceptStates[21] = 1;

	// T13 h14 e15
	experimentalNFA->maxState += 2;

	std::string books[] = {"romeo_and_juliet", "kafka", "tale_of_two_cities", "war_and_peace"};
	for (auto book : books) {
		std::string book_path = "D:/CUDFA/CUDFA/x64/Debug/" + book + ".txt";
		std::cout << book <<  " = " << runExperiment(book_path.c_str(), experimentalNFA) << std::endl;
	}
	
	getchar();
	return 0;


}






