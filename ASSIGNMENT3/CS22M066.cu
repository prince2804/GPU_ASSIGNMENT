/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/

// kernel for counting active indegree 
__global__ void count_aid(int *d_li,int *d_le,int *d_maxi,int *d_active,int *d_offset,int *d_csrlist,int *d_aid){
   int tid=blockIdx.x * blockDim.x + threadIdx.x;
    if(tid>=*d_li && tid<=*d_le){
        int index=d_offset[tid];
        int lastindex=d_offset[tid+1];
        if(d_active[tid]==1){
            for(int i=index;i<lastindex;i++){
                atomicAdd(&d_aid[d_csrlist[i]], 1);
                atomicMax(d_maxi, d_csrlist[i]);
            }
        }
    }
   
}

//kernel for activation of nodes
__global__ void activation(int *d_li,int * d_le,int *d_aid,int*d_apr, int *d_active){
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    if(tid>=*d_li && tid<=*d_le){
        if(d_aid[tid]>=d_apr[tid])
        {
            d_active[tid]=1;
        }
    }
}       

// kernel for deactivation of nodes
__global__ void deactivation(int *d_li,int*d_le,int *d_active,int * d_activeVertex){
    
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    if(tid>*d_li && tid<*d_le)
    {
        if(d_active[tid-1]==0 && d_active[tid+1]==0){
             d_active[tid]=0;
        }
    }
}

// kernel for counting active vertex at level
__global__ void count_activenode(int *d_li,int*d_le,int *d_active,int * d_activeVertex,int level){
    
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
     if(tid>=*d_li && tid<=*d_le){
        if(d_active[tid]==1){
            atomicAdd(&d_activeVertex[level], 1);
        }
    } 
}    


/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

     //variable for result, storing number of active vertices at each level, on device
     int *d_activeVertex;
	  cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

//Initialize d_aid array to zero for each vertex
cudaMemset(&d_aid, 0, V*sizeof(int));

// Initialize active array which Intialize zero for each vertex
int *h_active = (int*)malloc(V*sizeof(int));
memset(h_active, 0, V*sizeof(int));

int *d_active;
cudaMalloc(&d_active, V*sizeof(int));

/*for zero level vertex tarverse apr array and update number of activeVertex at level 0 and 
    active vertex by updating active array*/
int i;
for(i=0;i<V;i++)
{
    if(h_apr[i]==0) {
        h_activeVertex[0]++;
        h_active[i]=1;
    }
    else break;
}

//copy h_active and h_activeVertex to device array d_active and d_activeVertex 
cudaMemcpy(d_active, h_active, V*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_activeVertex, h_activeVertex, L*sizeof(int), cudaMemcpyHostToDevice);

//variable li=level intialize and le=level end and maxi=level maximum node 
int li=0,le=i-1;
int maxi=0;
int *d_maxi,*d_li,*d_le;

// allocate memory to device variable
cudaMalloc(&d_maxi, sizeof(int));
cudaMalloc(&d_li, sizeof(int));
cudaMalloc(&d_le, sizeof(int));

// level variable which contain current level intialize to 1 because we alredy traverse level 0 
int level=1;

cudaMemcpy(d_li, &li, sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_le, &le, sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_maxi, &maxi, sizeof(int), cudaMemcpyHostToDevice);

// this while loop traverse graph level by level
while(level<L)
{
    //le maximum vertex number at current level
    int nblocks=ceil((float)le/1024);

    //kernel launch for count active indegree
    count_aid<<<nblocks,1024>>>(d_li,d_le,d_maxi,d_active,d_offset,d_csrList,d_aid);
    cudaDeviceSynchronize();

    // now we update level intialize by 1 to last level end 
    li=le+1;

    // and we get maxi by previous level traverse(means maximum of last node in next level) 
    cudaMemcpy(&maxi, d_maxi, sizeof(int), cudaMemcpyDeviceToHost);
    le=maxi;

    nblocks=ceil((float)le/1024);
    cudaMemcpy(d_li, &li, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_le, &le, sizeof(int), cudaMemcpyHostToDevice);

    //kernel launch for activation of node 
    activation<<<nblocks,1024>>>(d_li,d_le,d_aid,d_apr,d_active); 

    //kernel launch for deactivation of node 
    deactivation<<<nblocks,1024>>>(d_li,d_le,d_active,d_activeVertex);

    //kernel for counting active node at level
    count_activenode<<<nblocks,1024>>>(d_li,d_le,d_active,d_activeVertex,level);

    //increase level
    level++;
}
 
//copy d_activeVertex to h_activeVertex
cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
