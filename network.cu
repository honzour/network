/*
@author Jan Nemec, jnem6403@seznam.cz

*/


#include <stdio.h>
#include <stdlib.h>

/*
	Global settings
*/

/** Compile as emulation or use CUDA */
#define EMULATION 0

/** Number of non input and non output groups of neuron */
#define HIDDEN_GROUPS 5

#define GROUP_COUNT (HIDDEN_GROUPS + 2)

/** Number of neuron in each group */

#define NEURONS_IN_GROUP 100

/** Divide each float coef by this */
#define DIVIDE_COEF 8192

/** bigger TRESHOLD_RAND -> bigger tresholds */
#define TRESHOLD_RAND 32768

/** maximal number of external connections */
#define MAX_EXTERNAL_CONNECTIONS 8

/** bigger WEIGHT_RAND -> bigger weights */
#define WEIGHT_RAND 110

/** bigger INPUT_RAND -> bigger input in the input layer */
#define INPUT_RAND 256

/*
	Global types
*/

/** we will compute in this type */
typedef float FLOAT_TYPE;

/**
	Data inside of each group (without connections to other groups)
*/
typedef struct
{
	/* full matrix NEURONS_IN_GROUP * NEURONS_IN_GROUP
	   weight from 1 to 2 is in w[1 + 2 * NEURONS_IN_GROUP]  */
	FLOAT_TYPE w[NEURONS_IN_GROUP * NEURONS_IN_GROUP];
	/*	0 .. NEURONS_IN_GROUP
		Fixed input (addes every step to potential of the neuron) */
	FLOAT_TYPE inputs[NEURONS_IN_GROUP];
	/* 0 .. NEURONS_IN_GROUP */
	FLOAT_TYPE tresholds[NEURONS_IN_GROUP];
	/* 0 .. NEURONS_IN_GROUP */
	FLOAT_TYPE potentials[NEURONS_IN_GROUP];
	/* is each neuron active in the curren step */
	unsigned char active[NEURONS_IN_GROUP]; 
} TGroupInternal;

/**
 Connection between groups
 */
typedef struct
{
	/** from group index */
	int group;
	/** from neuron (inside of the group) index */
	int neuron;
	/** weight */
	FLOAT_TYPE w;
} TConnection;

/** Group inluding without connections to other groups */
typedef struct
{
	/** Internal group data */
	TGroupInternal inside;
	/** Connections from another group
	   connections[1][2] is the third (0, 1, 2) connection of the second (0, 1)
       neuron */
	TConnection connections[NEURONS_IN_GROUP][MAX_EXTERNAL_CONNECTIONS];
	int connectionCount[NEURONS_IN_GROUP];
} TGroup;

/**
 All the network.
 */
typedef struct
{
	/** All groups, the first is input, the last is output. */
	TGroup groups[GROUP_COUNT];
} TNetwork;

/*
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}
*/

/**
	Init a single group of neurons
 	@param index index of this group
	@param group group to init
 */ 
void initGroup(int index, TGroup *group)
{
	int i;

	for (i = 0; i < NEURONS_IN_GROUP; i++)
	{
		int j;

		/* init connections from other groups */
		group->connectionCount[i] = rand() % MAX_EXTERNAL_CONNECTIONS;
		for (j = 0; j < group->connectionCount[i]; j++)
		{
			TConnection *conn = group->connections[i] + j;
			conn->group = rand() % GROUP_COUNT;
			conn->neuron = rand() % NEURONS_IN_GROUP;
			conn->w = (rand() % WEIGHT_RAND) / (FLOAT_TYPE) DIVIDE_COEF;
		}
	}
	/* init connections inside this group */
	
 	for (i = 0; i < NEURONS_IN_GROUP * NEURONS_IN_GROUP; i++)
	{
		group->inside.w[i] = (rand() % WEIGHT_RAND) / (FLOAT_TYPE) DIVIDE_COEF;
	}
	/* init all the data for each neuron */
	
	for (i = 0; i < NEURONS_IN_GROUP; i++)
	{
		group->inside.inputs[i] = index ? 0 : ((rand()  % INPUT_RAND) /
			(FLOAT_TYPE) DIVIDE_COEF);
		group->inside.tresholds[i] = (1 + (rand()  % TRESHOLD_RAND)) /
			(FLOAT_TYPE) DIVIDE_COEF;
		group->inside.potentials[i] = 0;
		group->inside.active[i] = 0;
	}
}

/**
 Inits every single group of the network. 
 */
void initNetwork(TNetwork *net)
{
	int i;

	for (i = 0; i < GROUP_COUNT; i++)
	{
		initGroup(i, net->groups + i);
	}
}

#if EMULATION

/**
 Single step of the computing
 */
void step(TNetwork *net)
{
	int i;

	/* The first step - connections from other group */

	/* for each group */
	for (i = 0; i < GROUP_COUNT; i++)
	{
		int j;

		TGroup *group = net->groups + i;
		/* for each neuron in the group */
		for (j = 0; j < NEURONS_IN_GROUP; j++)
		{
			int k;
			/* for each connection (from the other group) of the neuron */
			for (k = 0; k < group->connectionCount[j]; k++)
			{
				TConnection *conn = group->connections[j] + k;
				/* if the other neuron is active*/
				if (net->groups[conn->group].inside.active[conn->neuron])
				{
					/* add a bonus to our potential */
					group->inside.potentials[j] += conn->w;
				}
			}
		}
	}

	/* The second step */

	/* for each group */
	for (i = 0; i < GROUP_COUNT; i++)
	{
		int j, k;
		TGroup *group = net->groups + i;

		/* for each neuron in the group */
		for (j = 0; j < NEURONS_IN_GROUP; j++)
		{

			FLOAT_TYPE *ptrW = group->inside.w + j * NEURONS_IN_GROUP;
			unsigned char *ptrA = group->inside.active;

			/* for each connection */
			for (k = 0; k < NEURONS_IN_GROUP; k++)
			{
				if (*ptrA)
				{
					/* add the weight if the neuron is active */
					group->inside.potentials[j] += *ptrW;
				} 
				ptrW++;
				ptrA++;
			}
			/* Add input to the potential */ 
			group->inside.potentials[j] += group->inside.inputs[j];
		}
	}

	/* for each group */
	for (i = 0; i < GROUP_COUNT; i++)
	{
		int j;
		TGroup *group = net->groups + i;

		/* for each neuron in the group */
		for (j = 0; j < NEURONS_IN_GROUP; j++)
		{
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

/* print the output of thenetwork */
void printResult(TNetwork *net)
{
	int i;
	TGroup *last = net->groups + (GROUP_COUNT - 1);
	for (i = 0; i < NEURONS_IN_GROUP; i++)
	{
		putchar(last->inside.active[i] ? '1' : '0');
	}
	puts("");
}

#else

__global__ void updatePotentials(TNetwork *net)
{
	int g = threadIdx.x;
    int n = threadIdx.y;

	TGroup *group = net->groups + g;
	/* for each neuron in the group */
	
	int k;
	/* for each connection (from the other group) of the neuron */
	for (k = 0; k < group->connectionCount[n]; k++)
	{
		TConnection *conn = group->connections[n] + k;
		/* if the other neuron is active*/
		if (net->groups[conn->group].inside.active[conn->neuron])
		{
			/* add a bonus to our potential */
			group->inside.potentials[n] += conn->w;
		}
	}

	FLOAT_TYPE *ptrW = group->inside.w + n * NEURONS_IN_GROUP;
	unsigned char *ptrA = group->inside.active;

	/* for each connection */
	for (k = 0; k < NEURONS_IN_GROUP; k++)
	{
		if (*ptrA)
		{
			/* add the weight if the neuron is active */
			group->inside.potentials[n] += *ptrW;
		} 
		ptrW++;
		ptrA++;
	}
	/* Add input to the potential */ 
	group->inside.potentials[n] += group->inside.inputs[n];
}

__global__ void updateActive(TNetwork *net)
{
	int g = threadIdx.x;
    int n = threadIdx.y;

	TGroup *group = net->groups + g;
 
	if (group->inside.potentials[n] >= group->inside.tresholds[n])
	{
		group->inside.potentials[n] = 0;
		group->inside.active[n] = 1;
	}
	else
	{
		group->inside.active[n] = 0;
	}
}

/* print the output of thenetwork */
void printResult(TNetwork *d_net)
{

	/* TODO do not copy all !!! */
	TNetwork net;
	cudaMemcpy(&net, d_net, sizeof(TNetwork), cudaMemcpyDeviceToHost);

	int i;
	TGroup *last = net.groups + (GROUP_COUNT - 1);
	for (i = 0; i < NEURONS_IN_GROUP; i++)
	{
		putchar(last->inside.active[i] ? '1' : '0');
	}
	puts("");
}
#endif


int main(void)
{
	int i;
	TNetwork *net = (TNetwork *)malloc(sizeof(TNetwork));
	srand(time(NULL));
	initNetwork(net);

#if EMULATION
	for (i = 0; i < 1000; i++)
	{
		step(net);
		printResult(net);
	}
#else
	
	TNetwork *d_net;
	cudaMalloc(&d_net, sizeof(TNetwork));
	cudaMemcpy(d_net, net, sizeof(TNetwork), cudaMemcpyHostToDevice); 
	for (i = 0; i < 1000; i++)
	{
		updatePotentials<<<GROUP_COUNT, NEURONS_IN_GROUP>>>(d_net);
		updateActive<<<GROUP_COUNT, NEURONS_IN_GROUP>>>(d_net);
		printResult(d_net);
	}
	cudaFree(d_net);
#endif

	free(net);
	return 0;
}
