#include<iostream>
#include<sys/time.h>
#include<cuda.h>
#define BLOCKSIZE 1024
using namespace std;

//kernel for computing matrix A * matrix B
__global__ void ab_kernel(int p, int q, int r, int *d_matrixA, int *d_matrixB, int *d_AB){
           
            __shared__ int ab[BLOCKSIZE];

              int x=blockIdx.x;
              int y=blockIdx.y;

              int k=threadIdx.x;
              d_AB[r*x+y]=0;
              if(k<q)
                ab[k]=d_matrixA[q*x+k] * d_matrixB[k*r+y];
              __syncthreads();

              for(int i=0;i<q;i++)
                {
                    d_AB[r*x+y]=d_AB[r*x+y]+ab[i];
                }
}

//kernel for computing matrix C * matrix DT
__global__ void cdt_kernel(int p, int q, int r, int *d_matrixC, int *d_matrixD, int *d_CDT){
    
            __shared__ int cdt[BLOCKSIZE];

              int x=blockIdx.x;
              int y=blockIdx.y;

              int k=threadIdx.x;
              d_CDT[r*x+y]=0;
              if(k<q)
                cdt[k]=d_matrixC[q*x+k] * d_matrixD[q*y+k];
              __syncthreads();

              for(int i=0;i<q;i++)
                {
                    d_CDT[r*x+y]=d_CDT[r*x+y]+cdt[i];
                }
}

//kernel for computing matrix E

__global__ void sum_kernel(int p, int q, int r, int *matrixAB, int *matrixCDT, int *d_matrixE){
    
              int x=blockIdx.x;
              int y=blockIdx.y;


              d_matrixE[r*x+y]=matrixAB[r*x+y]+matrixCDT[r*x+y];

}


// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */

	int *matrixAB,*d_AB;
    matrixAB = (int*) malloc(p * r * sizeof(int));
   
    cudaMalloc(&d_AB, p * r * sizeof(int));
  
    dim3 grid(p,r,1);
    dim3 block(q,1,1);

    ab_kernel<<<grid,block>>>(p,q,r,d_matrixA,d_matrixB,d_AB);
    cudaDeviceSynchronize(); 

    cudaMemcpy(matrixAB ,d_AB, p * r * sizeof(int), cudaMemcpyDeviceToHost);

  /*------------------------------------------------------------------- */

   int *matrixCDT,*d_CDT;
   matrixCDT = (int*) malloc(p * r * sizeof(int));
   
   cudaMalloc(&d_CDT, p * r * sizeof(int));
  
   cdt_kernel<<<grid,block>>>(p,q,r,d_matrixC,d_matrixD,d_CDT);
   cudaDeviceSynchronize(); 

   cudaMemcpy(matrixCDT ,d_CDT, p * r * sizeof(int), cudaMemcpyDeviceToHost);

  /*------------------------------------------------------------------- */

    sum_kernel<<<grid,1>>>(p,q,r,d_AB,d_CDT,d_matrixE);
    cudaDeviceSynchronize(); 


 /* ****************************************************************** */

  	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}

