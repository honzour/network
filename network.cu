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

/** Group of neurons */
typedef struct
{
	/* full matrix NEURONS_IN_GROUP * NEURONS_IN_GROUP
	   weight from 1 to 2 is in w[group][1 + 2 * NEURONS_IN_GROUP]  */
	FLOAT_TYPE w[GROUP_COUNT][NEURONS_IN_GROUP * NEURONS_IN_GROUP];
	/*	0 .. NEURONS_IN_GROUP
		Fixed input (addes every step to potential of the neuron) */
	FLOAT_TYPE inputs[GROUP_COUNT][NEURONS_IN_GROUP];
	/* 0 .. NEURONS_IN_GROUP */
	FLOAT_TYPE tresholds[GROUP_COUNT][NEURONS_IN_GROUP];
	/* 0 .. NEURONS_IN_GROUP */
	FLOAT_TYPE potentials[GROUP_COUNT][NEURONS_IN_GROUP];
	/* is each neuron active in the curren step */
	unsigned char active[GROUP_COUNT][NEURONS_IN_GROUP]; 
	/** Connections from another group
	   connections_xx[group][1][2] is the third (0, 1, 2) connection of the second (0, 1)
       neuron */
	int connection_group[GROUP_COUNT][NEURONS_IN_GROUP][MAX_EXTERNAL_CONNECTIONS];
	int connection_neuron[GROUP_COUNT][NEURONS_IN_GROUP][MAX_EXTERNAL_CONNECTIONS];
	FLOAT_TYPE connection_w[GROUP_COUNT][NEURONS_IN_GROUP][MAX_EXTERNAL_CONNECTIONS];
	/** number of external connections */
	int connectionCount[GROUP_COUNT][NEURONS_IN_GROUP];
} TNetwork;

/**
 Inits every single group of the network. 
 */
void initNetwork(TNetwork *net)
{
	int group;

	for (group = 0; group < GROUP_COUNT; group++)
	{
		int i;

		for (i = 0; i < NEURONS_IN_GROUP; i++)
		{
			int j;

			/* init connections from other groups */
			net->connectionCount[group][i] = rand() % MAX_EXTERNAL_CONNECTIONS;
			for (j = 0; j < net->connectionCount[group][i]; j++)
			{
				net->connection_group[group][i][j] = rand() % GROUP_COUNT;
				net->connection_neuron[group][i][j] = rand() % NEURONS_IN_GROUP;
				net->connection_w[group][i][j] = ((rand() % WEIGHT_RAND) / (FLOAT_TYPE) DIVIDE_COEF);
			}
		}
		/* init connections inside this group */
	
	 	for (i = 0; i < NEURONS_IN_GROUP * NEURONS_IN_GROUP; i++)
		{
			net->w[group][i] = (rand() % WEIGHT_RAND) / (FLOAT_TYPE) DIVIDE_COEF;
		}
		/* init all the data for each neuron */
	
		for (i = 0; i < NEURONS_IN_GROUP; i++)
		{
			net->inputs[group][i] = group ? 0 :
				/* "normal" distribution to get more stable result */
				(
					(rand()  % INPUT_RAND) + (rand()  % INPUT_RAND) + 
					(rand()  % INPUT_RAND) + (rand()  % INPUT_RAND)
				) /	(FLOAT_TYPE) (DIVIDE_COEF * 4);
			net->tresholds[group][i] = (1 + (rand()  % TRESHOLD_RAND)) /
				(FLOAT_TYPE) DIVIDE_COEF;
			net->potentials[group][i] = 0;
			net->active[group][i] = 0;
		}
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

		/* for each neuron in the group */
		for (j = 0; j < NEURONS_IN_GROUP; j++)
		{
			int k;
			/* for each connection (from the other group) of the neuron */
			for (k = 0; k < net->connectionCount[i][j]; k++)
			{
				/* if the other neuron is active*/
				if (
				    net->active
				    	[ net->connection_group[i][j][k] ]
						[ net->connection_neuron[i][j][k] ]
				    )
				{
					/* add a bonus to our potential */
					net->potentials[i][j] += net->connection_w[i][j][k];
				}
			}
		}
	}

	/* The second step */

	/* for each group */
	for (i = 0; i < GROUP_COUNT; i++)
	{
		int j, k;

		/* for each neuron in the group */
		for (j = 0; j < NEURONS_IN_GROUP; j++)
		{

			FLOAT_TYPE *ptrW = net->w[i] + j * NEURONS_IN_GROUP;
			unsigned char *ptrA = net->active[i];

			/* for each connection */
			for (k = 0; k < NEURONS_IN_GROUP; k++)
			{
				if (*ptrA)
				{
					/* add the weight if the neuron is active */
					net->potentials[i][j] += *ptrW;
				} 
				ptrW++;
				ptrA++;
			}
			/* Add input to the potential */ 
			net->potentials[i][j] += net->inputs[i][j];
		}
	}

	/* for each group */
	for (i = 0; i < GROUP_COUNT; i++)
	{
		int j;

		/* for each neuron in the group */
		for (j = 0; j < NEURONS_IN_GROUP; j++)
		{
		/* Check tresholds and set active neuron*/
			if (net->potentials[i][j] >= net->tresholds[i][j])
			{
				net->potentials[i][j] = 0;
				net->active[i][j] = 1;
			}
			else
			{
				net->active[i][j] = 0;
			}
		}
	}
}

/* print the output of the network */
void printResult(int line, TNetwork *net)
{
	printOutputArray(line, net->active[GROUP_COUNT - 1]);
}

#else

/**
	One step of computing - updating of potentials
*/
__global__ void updatePotentials(TNetwork *net)
{
	int g = blockIdx.x;
	int n = threadIdx.x;
	
	int k;
	/* for each connection (from the other group) of the neuron */
	for (k = 0; k < net->connectionCount[g][n]; k++)
	{
		if (
		    net->active
		    	[ net->connection_group[g][n][k] ]
				[ net->connection_neuron[g][n][k] ]
		    )
		{
			/* add a bonus to our potential */
			net->potentials[g][n] += net->connection_w[g][n][k];
		}
	}

	FLOAT_TYPE *ptrW = net->w[g] + n * NEURONS_IN_GROUP;
	unsigned char *ptrA = net->active[g];

	/* for each connection */
	for (k = 0; k < NEURONS_IN_GROUP; k++)
	{
		if (*ptrA)
		{
			/* add the weight if the neuron is active */
			net->potentials[g][n] += *ptrW;
		} 
		ptrW++;
		ptrA++;
	}
	/* Add input to the potential */ 
	net->potentials[g][n] += net->inputs[g][n];
}

/**
	One step of computing - updating of active states
*/
__global__ void updateActive(TNetwork *net)
{
	int g = blockIdx.x;
	int n = threadIdx.x;
 
	if (net->potentials[n] >= net->tresholds[n])
	{
		net->potentials[g][n] = 0;
		net->active[g][n] = 1;
	}
	else
	{
		net->active[g][n] = 0;
	}
}

/**
	Copy active states from the output group from the device memory
 	TODO - make this faster 
*/
__global__ void getOutput(TNetwork *net, unsigned char *output)
{
	int n = threadIdx.x;

	output[n] = net->active[GROUP_COUNT - 1][n];
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
