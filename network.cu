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
	/* full matrix NEURONS_IN_GROUP * NEURONS_IN_GROUP
	   weight from 1 to 2 is in w[1 + 2 * NEURONS_IN_GROUP]  */
	FLOAT_TYPE *w;
	/*	0 .. NEURONS_IN_GROUP
		Fixed input (addes every step to potential of the neuron) */
	FLOAT_TYPE *inputs;
	/* 0 .. NEURONS_IN_GROUP */
	FLOAT_TYPE *tresholds;
	/* 0 .. NEURONS_IN_GROUP */
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
	/* Connections from another group
	   connections[1][2] is the third (0, 1, 2) connection of the second (0, 1)
       neuron */
	TConnection **connections;
	int *connectionCount;
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

void initGroup(int hiddenGroups, int neuronsInGroup, int index, TGroup *group)
{
	int i;

	group->connections = (TConnection **) malloc(sizeof(TConnection *) * 
		neuronsInGroup);
	group->connectionCount = (int *) malloc(sizeof(int) * neuronsInGroup);
	for (i = 0; i < neuronsInGroup; i++)
	{
		int j;

		/* init connections from other groups */
		group->connectionCount[i] = rand() % 8;
		group->connections[i] = (TConnection *)
			malloc(sizeof(TConnection) * group->connectionCount[i]);
		for (j = 0; j < group->connectionCount[i]; j++)
		{
			TConnection *conn = group->connections[i] + j;
			conn->group = rand() % (hiddenGroups + 2);
			conn->neuron = rand() % neuronsInGroup;
			conn->w = (rand() & 0xFF) / (FLOAT_TYPE)512.0;
		}
	}
	/* init connections inside this group */
	group->inside.w = (FLOAT_TYPE *) malloc(sizeof(FLOAT_TYPE) *
		neuronsInGroup * neuronsInGroup);
 	for (i = 0; i < neuronsInGroup * neuronsInGroup; i++)
	{
		group->inside.w[i] = (rand() & 0xFF) / (FLOAT_TYPE)512.0;
	}
	/* init all the data for each neuron */
	group->inside.inputs = (FLOAT_TYPE *) malloc(sizeof(FLOAT_TYPE) *
		neuronsInGroup);
	group->inside.tresholds = (FLOAT_TYPE *) malloc(sizeof(FLOAT_TYPE) *
		neuronsInGroup);
	group->inside.potentials = (FLOAT_TYPE *) malloc(sizeof(FLOAT_TYPE) *
		neuronsInGroup);
	group->inside.active = (unsigned char *) malloc(neuronsInGroup);
	for (i = 0; i < neuronsInGroup; i++)
	{
		group->inside.inputs[i] = (i ? 0 : (rand() & 1));
		group->inside.tresholds[i] = (rand() & 0xFF) / (FLOAT_TYPE) 128.0;
		group->inside.potentials[i] = 0;
		group->inside.active[i] = 0;
	}
}

void doneGroup(int neuronsInGroup, TGroup *group)
{
	int i;
	for (i = 0; i < neuronsInGroup; i++)
	{
		free(group->connections[i]);
	}
	free(group->connections);
	free(group->connectionCount);
	free(group->inside.inputs);
	free(group->inside.tresholds);
	free(group->inside.potentials);
	free(group->inside.active);
}

void initNetwork(int hiddenGroups, int neuronsInGroup, TNetwork *net)
{
	int i;
	int limit = hiddenGroups + 2;
	net->groupCount = limit;
	net->groups = (TGroup *) malloc(sizeof(TGroup) * limit);
	for (i = 0; i < limit; i++)
	{
		initGroup(hiddenGroups, neuronsInGroup, i, net->groups + i);
	}
}

void doneNetwork(int neuronsInGroup, TNetwork *net)
{
	int i;
	for (i = 0; i < net->groupCount; i++)
	{
		doneGroup(neuronsInGroup, net->groups + i);
	}
	free(net->groups);
}

void step(TNetwork *net)
{
	int i;

	/* The first step which is hard to make paralell 
		- connections from other group, it is hard to separate the memory */

	for (i = 0; i < net->groupCount; i++)
	{
		int j;

		TGroup *group = net->groups + i;
		for (j = 0; j < NEURONS_IN_GROUP; j++)
		{
			int k;
			for (k = 0; k < group->connectionCount[j]; k++)
			{
				TConnection *conn = group->connections[j] + k;
				if (net->groups[conn->group].inside.active[conn->neuron])
				{
					group->inside.potentials[j] += conn->w;
				}
			}
		}
	}

	/* The second step should be done paralell */
	for (i = 0; i < net->groupCount; i++)
	{
		int j, k;
		TGroup *group = net->groups + i;
	
		for (j = 0; j < NEURONS_IN_GROUP; j++)
		{
			/* Add bonus from connection inside of the group to potential of each
		   	   neuron. */

			FLOAT_TYPE *ptrW = group->inside.w + j * NEURONS_IN_GROUP;
			unsigned char *ptrA = group->inside.active;
			for (k = 0; k < NEURONS_IN_GROUP; k++)
			{
				if (*ptrA)
				{
					group->inside.potentials[j] += *ptrW;
				} 
				ptrW++;
				ptrA++;
			}
			/* Add input to the potential */ 
			group->inside.potentials[j] += group->inside.inputs[j];

			/* Check tresholds and set active neuron*/
			if (group->inside.potentials[j] >= group->inside.tresholds[j])
			{
				group->inside.potentials[j] = 0;
				group->inside.active[j] = 1;
			}
			else
			{
				group->inside.active[j] = 0;
			}
		}
	}
}

void printResult(TNetwork *net)
{
	int i;
	TGroup *last = net->groups + (net->groupCount - 1);
	for (i = 0; i < NEURONS_IN_GROUP; i++)
	{
		putchar(last->inside.active[i] ? '1' : '0');
	}
	puts("");
}


int main(void)
{
	int i;
	TNetwork net;
	srand(time(NULL));
	initNetwork(HIDDEN_GROUPS, NEURONS_IN_GROUP, &net);
	for (i = 0; i < 10; i++)
	{
		step(&net);
		printResult(&net);
	}
	doneNetwork(NEURONS_IN_GROUP, &net);
	/* VecAdd<<<1, N>>>(A, B, C); */

	return 0;
}
