#include <stdio.h>

/*
	Global settings
*/

int HIDDEN_LAYERS = 5;
int NEURONS_IN_LAYER = 100;
int EMULATION = 1;

/*
	Global types
*/

typedef FLOAT_TYPE float;

/*
	Data inside of each group (without connections to other groups)
*/
typedef struct
{
	/* full matrix NEURONS_IN_LAYER * NEURONS_IN_LAYER
	   weight from 1 to 2 is in w[1 + 2 * NEURONS_IN_LAYER] */
	FLOAT_TYPE *w;
	/* 0 .. NEURONS_IN_LAYER */
	FLOAT_TYPE *tresholds;
	/* 0 .. NEURONS_IN_LAYER */
	FLOAT_TYPE *potentials;
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
}


// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}



int main(void)
{
	float A[N];
	float B[N];
	float C[N];

	int i;

	for (i = 0; i < N; i++)
	{
		A[i]=i; B[i]=10;
	}

	VecAdd<<<1, N>>>(A, B, C);

	printf("je to %i\n", (int)A[5]);

	return 0;
}
