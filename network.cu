/*
@author Jan Nemec, jnem6403@seznam.cz

*/


#include <stdio.h>
#include <stdlib.h>

/*
	Global settings
*/

/** Compile as emulation or use CUDA */
#define EMULATION 1

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

/** how many steps to copmpute */
#define ITERATIONS 1000

/*
	Global types
*/

/** we will compute in this type */
typedef float FLOAT_TYPE;

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

/** Group of neurons */
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
		group->w[i] = (rand() % WEIGHT_RAND) / (FLOAT_TYPE) DIVIDE_COEF;
	}
	/* init all the data for each neuron */
	
	for (i = 0; i < NEURONS_IN_GROUP; i++)
	{
		group->inputs[i] = index ? 0 :
			/* "normal" distribution to get more stable result */
			(
				(rand()  % INPUT_RAND) + (rand()  % INPUT_RAND) + 
				(rand()  % INPUT_RAND) + (rand()  % INPUT_RAND)
			) /	(FLOAT_TYPE) (DIVIDE_COEF * 4);
		group->tresholds[i] = (1 + (rand()  % TRESHOLD_RAND)) /
			(FLOAT_TYPE) DIVIDE_COEF;
		group->potentials[i] = 0;
		group->active[i] = 0;
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

/* print the sinle line of the output */
void printOutputArray(int line, const unsigned char *output)
{
	int i;

	printf("%i ", line);
	for (i = 0; i < NEURONS_IN_GROUP; i++)
	{
		putchar(output[i] ? '1' : '0');
	}
	puts("");
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
				if (net->groups[conn->group].active[conn->neuron])
				{
					/* add a bonus to our potential */
					group->potentials[j] += conn->w;
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

			FLOAT_TYPE *ptrW = group->w + j * NEURONS_IN_GROUP;
			unsigned char *ptrA = group->active;

			/* for each connection */
			for (k = 0; k < NEURONS_IN_GROUP; k++)
			{
				if (*ptrA)
				{
					/* add the weight if the neuron is active */
					group->potentials[j] += *ptrW;
				} 
				ptrW++;
				ptrA++;
			}
			/* Add input to the potential */ 
			group->potentials[j] += group->inputs[j];
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
			if (group->potentials[j] >= group->tresholds[j])
			{
				group->potentials[j] = 0;
				group->active[j] = 1;
			}
			else
			{
				group->active[j] = 0;
			}
		}
	}
}

/* print the output of the network */
void printResult(int line, TNetwork *net)
{
	TGroup *last = net->groups + (GROUP_COUNT - 1);
	printOutputArray(line, last->active);
}

#else

/**
	One step of computing - updating of potentials
*/
__global__ void updatePotentials(TNetwork *net)
{
	int g = blockIdx.x;
	int n = threadIdx.x;

	TGroup *group = net->groups + g;
	/* for each neuron in the group */
	
	int k;
	/* for each connection (from the other group) of the neuron */
	for (k = 0; k < group->connectionCount[n]; k++)
	{
		TConnection *conn = group->connections[n] + k;
		/* if the other neuron is active*/
		if (net->groups[conn->group].active[conn->neuron])
		{
			/* add a bonus to our potential */
			group->potentials[n] += conn->w;
		}
	}

	FLOAT_TYPE *ptrW = group->w + n * NEURONS_IN_GROUP;
	unsigned char *ptrA = group->active;

	/* for each connection */
	for (k = 0; k < NEURONS_IN_GROUP; k++)
	{
		if (*ptrA)
		{
			/* add the weight if the neuron is active */
			group->potentials[n] += *ptrW;
		} 
		ptrW++;
		ptrA++;
	}
	/* Add input to the potential */ 
	group->potentials[n] += group->inputs[n];
}

/**
	One step of computing - updating of active states
*/
__global__ void updateActive(TNetwork *net)
{
	int g = blockIdx.x;
	int n = threadIdx.x;

	TGroup *group = net->groups + g;
 
	if (group->potentials[n] >= group->tresholds[n])
	{
		group->potentials[n] = 0;
		group->active[n] = 1;
	}
	else
	{
		group->active[n] = 0;
	}
}

/**
	Copy active states from the output group from the device memory
*/
__global__ void getOutput(TNetwork *net, unsigned char *output)
{
	int n = threadIdx.x;

	output[n] = net->groups[GROUP_COUNT - 1].active[n];
}

/** report error and exit */
void handleError(cudaError_t e, const char *function)
{
	fprintf(stderr, "Error %u in %s (%s), exiting\n",
		(unsigned) e, function, cudaGetErrorString(e));
	exit(1);
}

/** check cudaGetLastError() */
void checkAndHandleKernelError(const char *function)
{
	cudaError_t e;
	e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		handleError(e, function);
	}
}

/** check the function call return code */
void checkAndHandleFunctionError(cudaError_t e, const char *function)
{
	if (e != cudaSuccess)
	{
		handleError(e, function);
	}
}

#endif


int main(void)
{
	int i;
	TNetwork *net = (TNetwork *)malloc(sizeof(TNetwork));
	srand(time(NULL));
	initNetwork(net);

#if EMULATION
	for (i = 0; i < ITERATIONS; i++)
	{
		step(net);
		printResult(i, net);
	}
#else
	
	TNetwork *d_net;

	checkAndHandleFunctionError(cudaMalloc(&d_net, sizeof(TNetwork)),
		"cudaMalloc");
	checkAndHandleFunctionError(cudaMemcpy(d_net, net, sizeof(TNetwork),
		cudaMemcpyHostToDevice), "cudaMemcpy"); 
	for (i = 0; i < ITERATIONS; i++)
	{
		unsigned char active[NEURONS_IN_GROUP];


		updatePotentials<<<GROUP_COUNT, NEURONS_IN_GROUP>>>(d_net);
		checkAndHandleKernelError("updatePotentials");
		
		updateActive<<<GROUP_COUNT, NEURONS_IN_GROUP>>>(d_net);
		checkAndHandleKernelError("updateActive");

		getOutput<<<1, NEURONS_IN_GROUP>>>(d_net, active);
		checkAndHandleKernelError("getOutput");

		printOutputArray(i, active);
	}
	checkAndHandleFunctionError(cudaFree(d_net), "cudaFree");
#endif

	free(net);
	return 0;
}
