#include <stdio.h>
#include <stdlib.h>

/*
	Global settings
*/

int HIDDEN_GROUPS = 5;
int NEURONS_IN_GROUP = 100;
int EMULATION = 1;

/*
	Global types
*/

typedef float FLOAT_TYPE;

/*
	Data inside of each group (without connections to other groups)
*/
typedef struct
{
	/* full matrix NEURONS_IN_LAYER * NEURONS_IN_LAYER
	   weight from 1 to 2 is in w[1 + 2 * NEURONS_IN_LAYER]  */
	FLOAT_TYPE *w;
	/*	0 .. NEURONS_IN_LAYER 
		Fixed input (addes every step to potential of the neuron) */
	FLOAT_TYPE *inputs;
	/* 0 .. NEURONS_IN_LAYER */
	FLOAT_TYPE *tresholds;
	/* 0 .. NEURONS_IN_LAYER */
	FLOAT_TYPE *potentials;
	/* is each neuron active in the curren step */
	unsigned char *active; 
} TGroupInternal;

/* Connection between groups*/
typedef struct
{
	/* from group index */
	int group;
	/* from neuron (inside of the group) index */
	int neuron;
	/* weight */
	FLOAT_TYPE w;
} TConnection;

/* Group inluding without connections to other groups */
typedef struct
{
	/* Internal group data */
	TGroupInternal inside;
	/* Connection inside the group
	   connections[1][2] is the third (0, 1, 2) connection of the second (0, 1)
       neuron */
	TConnection **connections;
} TGroup;

typedef struct
{
	/* All groups, the first is input, the last is output. */
	TGroup *groups;
	/* Including input and output */
	int groupCount;
} TNetwork;


// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

void initGroup(int hiddenGroups, int neuronsInGroup, TGroup *group)
{
	
}

void doneGroup(TGroup *group)
{

}

void initNetwork(int hiddenGroups, int neuronsInGroup, TNetwork *net)
{
	int i;
	int limit = hiddenGroups + 2;
	net->groupCount = limit;
	net->groups = (TGroup *) malloc(sizeof(TGroup) * limit);
	for (i = 0; i < limit; i++)
	{
		initGroup(hiddenGroups, neuronsInGroup, net->groups + i);
	}
}

void doneNetwork(TNetwork *net)
{
	int i;
	for (i = 0; i < net->groupCount; i++)
	{
		doneGroup(net->groups + i);
	}
	free(net->groups);
}

int main(void)
{
	TNetwork net;
	srand(time(NULL));
	initNetwork(HIDDEN_GROUPS, NEURONS_IN_GROUP, &net);
	doneNetwork(&net);
	/* VecAdd<<<1, N>>>(A, B, C); */

	return 0;
}
