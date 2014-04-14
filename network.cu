#include <stdio.h>

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
int i = threadIdx.x;
C[i] = A[i] + B[i];
}

#define N 10

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
