#include <bits/stdc++.h>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************
struct request{
       int id;
       int cen;
       int fac;
       int start;
       int slots;
   };
bool compare(request a,request b)
    {
        if(a.cen!=b.cen) return a.cen<b.cen;
     
        if(a.fac!=b.fac) return a.fac<b.fac;

        return (a.id<b.id);
    }

// Write down the kernels here
__global__ void find_successreq(int *d_start,int *d_end,int *d_success,int *d_succreq,int *d_facility,int *d_capacity,int *d_reqid,int *d_reqcen,int *d_reqfac,int *d_reqstart,int *d_reqslots,int *d_prefixsum)
{ 
    int center=blockIdx.x;
    int tid=threadIdx.x;
    int ty[25];
    if(tid<d_facility[center])
    {
        
        for(int i=0;i<=24;i++) ty[i]=0;
        int x=0;
        if(center!=0) x=d_prefixsum[center-1];
        x+=tid;
        if(d_start[x]!=-1)  {
            
            for(int i=d_start[x];i<=d_end[x];i++){
                  bool clash=false;
                  for(int j=d_reqstart[i];j<d_reqstart[i]+d_reqslots[i];j++){
                        if(j>25) clash=true;
                        if(ty[j]>=d_capacity[x]){
                            clash=true;
                            break;
                        }
                  }
                  if(!clash)
                  {
                        atomicAdd(d_success,1);
                        atomicAdd(&(d_succreq[center]),1);
                        for(int j=d_reqstart[i];j<d_reqstart[i]+d_reqslots[i];j++){
                                ty[j]=ty[j]+1;
                          }
                  }
              
             }
        }

    }
}

//***********************************************


int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }
    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
    int R;
    fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
		 
 
    //*********************************
    // Call the kernels here
    //********************************
    int *d_centre,*d_facility,*d_facids,*d_capacity,*d_success,*d_succreq;
    cudaMalloc(&d_centre,N*sizeof(int));
    cudaMalloc(&d_facility,N*sizeof(int));
    cudaMalloc(&d_facids,max_P*N*sizeof(int));
    cudaMalloc(&d_capacity,max_P*N*sizeof(int));
 
    cudaMalloc(&d_success,sizeof(int));
    cudaMemset(&d_success,0,sizeof(int));
    cudaMalloc(&d_succreq,N*sizeof(int));
    cudaMemset(&d_succreq,0,N*sizeof(int));
 

    cudaMemcpy(d_centre,centre, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_facility,facility, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_facids,fac_ids, max_P * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity,capacity, max_P *N * sizeof(int), cudaMemcpyHostToDevice);
    

    int *d_reqid,*d_reqcen,*d_reqfac,*d_reqstart,*d_reqslots;
 
    cudaMalloc(&d_reqid,R*sizeof(int));
    cudaMalloc(&d_reqcen,R*sizeof(int));
    cudaMalloc(&d_reqfac,R*sizeof(int));
    cudaMalloc(&d_reqstart,R*sizeof(int)); 
    cudaMalloc(&d_reqslots,R*sizeof(int));
 

    request a[R];
    for(int i=0;i<R;i++)
    {
        a[i].id=req_id[i];
        a[i].cen=req_cen[i];
        a[i].fac=req_fac[i];
        a[i].start=req_start[i];
        a[i].slots=req_slots[i];
    }
    sort(a,a+R,compare);

   for(int i=0;i<R;i++)
    {
        req_id[i]=a[i].id;
        req_cen[i]=a[i].cen;
        req_fac[i]=a[i].fac;
        req_start[i]=a[i].start;
        req_slots[i]=a[i].slots;
    }
 
    cudaMemcpy(d_reqid,req_id, R * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reqcen,req_cen, R * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reqfac,req_fac, R * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reqstart,req_start, R * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reqslots,req_slots, R * sizeof(int), cudaMemcpyHostToDevice);
 
    int *prefix_sum,*d_prefixsum;
    prefix_sum=(int *) malloc ( (N) * sizeof (int) );
    cudaMalloc(&d_prefixsum,N*sizeof(int));
    int total=0;
    for(int i=0;i<N;i++)
    {
        total+=facility[i];
        prefix_sum[i]=total;
    }
    cudaMemcpy(d_prefixsum,prefix_sum,N*sizeof(int),cudaMemcpyHostToDevice);
    int i=0;
    int *start,*end;
    start = (int *) malloc ( N*max_P * sizeof (int) );
    end = (int *) malloc  ( N*max_P * sizeof (int) );
    memset(start,-1,N*max_P*sizeof(int));
    memset(end,-1,N*max_P*sizeof(int));
 
    while(i<R)
    {
        int t1=req_cen[i];
        int t2=req_fac[i];
        int x=0;
        if(t1!=0) x=prefix_sum[t1-1];
        x+=t2;
        int s=i;
        while(i<R-1 &&req_cen[i]==req_cen[i+1] && req_fac[i]==req_fac[i+1]){
            i++;
        }
        int e=i;
        start[x]=s,end[x]=e;
        i++;
    }
      int *d_start,*d_end;
      cudaMalloc(&d_start,N*max_P*sizeof(int));
      cudaMalloc(&d_end,N*max_P*sizeof(int));
 
    cudaMemcpy(d_start,start, N * max_P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end,end, N * max_P * sizeof(int), cudaMemcpyHostToDevice);
 
    find_successreq<<<N,max_P>>>(d_start,d_end,d_success,d_succreq,d_facility,d_capacity,d_reqid,d_reqcen,d_reqfac,d_reqstart,d_reqslots,d_prefixsum);
    cudaDeviceSynchronize();

    cudaMemcpy(&success,d_success,sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(succ_reqs,d_succreq,N*sizeof(int), cudaMemcpyDeviceToHost);
    fail=R-success;
 
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}