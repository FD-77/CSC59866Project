//nvcc -O2 pointindexing_template.cu -o pointindexing -I /home/microway/cuda-samples/Common/
//./pointindexing  100 2 
//./pointindexing  10000000 10  
//point now
#include <helper_functions.h>
#include <helper_cuda.h>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <iterator>
#include <sys/time.h>
#include <time.h>

typedef unsigned short ushort;
typedef unsigned char uchar;

using namespace std;
//so, we are getting two time value t1 and t0
//We calculate the mircseconds of each by multiplying the seconds by 1000000
//add the fromal calculation of the system of microsecond of each
//then subtracting the answwer of t1 -t0
//We store it into t and convert it into milliseconds
//finally we print it
float calc_time(char *msg,timeval t0, timeval t1)
{
 	long d = t1.tv_sec*1000000+t1.tv_usec - t0.tv_sec * 1000000-t0.tv_usec;
 	float t=(float)d/1000;
 	if(msg!=NULL)
 		printf("%s ...%10.3f\n",msg,t);
 	return t;
}
// We have three arguments err, *file, and line
//one is a cudaErro_t type, a const char, and int
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
//this  if  expression is to make our own error statements in terminal
//So it will only run when we have an error in compile
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
//My guess, is that this line will determine how our error will be displayed
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct point2d
{
//ushort is a unsigned 16-bit integer from 0-65,535
    ushort x,y;
//When you use host and device together for a variable, you  are making it a global variable for the host side and the device side.
//So, the point2d can be accessed by the host and device.
//This is basically the constructor
    __host__ __device__
    point2d() {}
//So this is same process. This will create an object that will hold _x and _y
    __host__ __device__
    point2d(ushort _x, ushort _y) : x(_x), y(_y) {}

};

struct xytor
{

  int lev;
//Make an object with an int parameter
  __host__ __device__
  xytor(int lev): lev(lev) {}
//Basically an call operator
    __host__ __device__
    uint operator()(point2d p )
    {//takes in an point2d object
	//it extracts the x and y coordinates
	    ushort x = p.x;
	    ushort y = p.y;
	    {//>> means it shifts the bits (in binary) 16-lev places to the right
		ushort a=x>>(16-lev);
		ushort b=y>>(16-lev);
		return b*(1<<lev)+a;//return a calculated index that represents both of the coordinates
	    }
    }
};

//To understand the whole program
//argc means the "argument count" - how many command-line arguments were passed to the program.
//argv means "argument vector" an array of char pointers holding those arguments
int main(int argc, char *argv[])
{
//To understand this bit, if the argument count is not 3, then terminate
//For example, when you execute "./program" this is 1 count = argv[0]
//"./program 1000" is two counts with argv[0], argv[1]
//"./program 1000 8" is three counts with argv[0], argv[1], argv[2]
    if (argc!=3)
    {
        printf("USAGE: %s #points lev(0<lev<16)\n", argv[0]);
        exit(1);
    }
	//converts the second argument count  from string to integer
//which means that the second argument count is responsible for the number of coordinates in the system
    int num_points = atoi(argv[1]);
//Just making sure the num of points is 10000
    if(num_points<0) num_points=10000;    
    printf("num_points=%d\n",num_points);

    //The third argument is the integer that will determine how many places it moves in the bits.
    int run_lev=atoi(argv[2]);
    if(run_lev<0) run_lev=10;    //testing with 100 points
    printf("run_lev=%d\n",run_lev);
//make a vector called da with length of the number of points we have
    vector<unsigned int> da(num_points);
//make variables type timeval
    timeval s0, s1,s2,s3,s4,s5,s6,s7;
    
    //allocate host memory
//a pointer called h_points that points to an array of type point2d elements
// h_points has length of num_points
    point2d * h_points=new point2d[num_points];
//we will now create another pointer called h_cellids of type unsigned integer(only positive numbers).
//it points to an array of lenght num_points
    uint *h_cellids=new uint[num_points];
//creating another pointer called dptr_points of type point2D
//only points to NULL;
    point2d * dptr_points=NULL;
//same thing but type uint (probably for the h_cellids
    uint *dptr_cellids=NULL;
    //reports an error if something is wrong
    //using cudaMalloc, it is allocating memory on the GPu devic
	//transferring data from the host(CPU) to the device(GPU)
//first off, what is cudaMalloc?
/*
cudaMalloc -> a CUDA runtime API function basically the same as the malloc() in C 
malloc() -> allocates a memory block of at least size bytes. The CPU uses this
Similarly, the cudaMalloc() is what the GPU uses on its device
cudaMalloc parameters - it use two parameters. One double (as you can see below) and one number
the double pointer is to store the address of the  memory on the GPU in the pointer
the number is to know how many bytes you wan to allocate.
*/
/*
With this information, the code below want to allocate memory or memory block on the GPU for
dptr_points and dptr_cellids.
Basically, memory for the type point2D points and memory for their cell Ids.
*/
    HANDLE_ERROR( cudaMalloc( (void**)&dptr_points,num_points* sizeof(point2d)));
    HANDLE_ERROR( cudaMalloc( (void**)&dptr_cellids,num_points* sizeof(uint)));
    assert(dptr_points!=NULL&&dptr_cellids!=NULL);
    //The comment basically explains it
    //generate points
    gettimeofday(&s0, NULL);    
    for(int i=0;i<num_points;i++)
    {
    	point2d p;
    	p.x=random()%65536;
    	p.y=random()%65536;
    	h_points[i]=p;
    }
// get the time of day and storing it in the address of s1 or the pointer of s1. We are using NULL because
//we don't want the timezone (which is what the second parameter is for).
    gettimeofday(&s1, NULL);    
    calc_time("generating random points\n",s0,s1);
    
    // copy point data from CPU host to GPU device
/*
cudaMemcpy() is responsible of copying data between CPU memory to GPU memory
It needs four parameters:
1 - a destination pointer
2 - a source pointer (basically the data you want to copy)
3 - number of bytes you want to copy
4 - which way are you copying it towards (CPU to GPU or GPU to CPU???)
*/
    HANDLE_ERROR( cudaMemcpy( dptr_points, h_points, num_points * sizeof(point2d), cudaMemcpyHostToDevice ) ); 
    gettimeofday(&s2, NULL);    
    calc_time("trnasferring data to GPU\n",s1,s2);

/*
Thrust is basically the the GPU version of C++ Standard template library
like vectors. Basically allowing all the functions vector on the GPU
*/
    thrust::device_ptr<point2d> d_points=thrust::device_pointer_cast(dptr_points);
    thrust::device_ptr<uint> d_cellids =thrust::device_pointer_cast(dptr_cellids);
    
    //====================================================================================================
    //YOUR WORK below: Step 1- transform point coordinates to cell identifiers; pay attention to functor xytor
    //thrust::transform(...);
// code:
    thrust::transform(d_points, d_points+num_points,d_cellids,xytor(run_lev)); 
    cudaDeviceSynchronize();
    gettimeofday(&s3, NULL);
    calc_time("transforming..............\n",s2,s3);   
    
    //YOUR WORK below: Step 2- sort (cellid,point) pairs 
    //thrust::stable_sort_by_key(...)
    thrust::stable_sort_by_key(d_cellids, d_cellids+num_points,d_points);
    cudaDeviceSynchronize();
    gettimeofday(&s4, NULL);
    calc_time("sorting..............\n",s3,s4);
    
    uint *dptr_PKey=NULL;
    uint *dptr_PLen=NULL;
    uint *dptr_PPos=NULL;    
    HANDLE_ERROR( cudaMalloc( (void**)&dptr_PKey,num_points* sizeof(uint)));
    HANDLE_ERROR( cudaMalloc( (void**)&dptr_PLen,num_points* sizeof(uint)));
    HANDLE_ERROR( cudaMalloc( (void**)&dptr_PPos,num_points* sizeof(uint)));
    assert(dptr_PKey!=NULL&&dptr_PLen!=NULL&&dptr_PPos!=NULL);
    thrust::device_ptr<uint> d_PKey =thrust::device_pointer_cast(dptr_PKey);
    thrust::device_ptr<uint> d_PLen=thrust::device_pointer_cast(dptr_PLen);
    thrust::device_ptr<uint> d_PPos=thrust::device_pointer_cast(dptr_PPos);
    
    //YOUR WORK below: Step 3- reduce by key 
    //use  d_cellids as the first input vector and thrust::constant_iterator<int>(1) as the second input
    size_t num_cells=0;//num_cells is initialized to 0 just to make the template compile; it should be updated next
    // num_cells = thrust::reduce_by_key(...).first - d_PKey 	
    auto data_of_reduce =thrust::reduce_by_key(d_cellids,d_cellids+num_points,thrust::constant_iterator<int>(1),d_PKey,d_PLen);
    num_cells= data_of_reduce.first - d_PKey;
    cudaDeviceSynchronize();
    gettimeofday(&s5, NULL);
    calc_time("reducing.......\n",s4,s5);
    
    //YOUR WORK below: Step 4-  exclusive scan using d_PLen as the input and d_PPos as the output
    //thrust::exclusive_scan(...)
    thrust:exclusive_scan(d_PLen, d_PLen+num_points, d_PPos);
    cudaDeviceSynchronize();
    gettimeofday(&s6, NULL);
    calc_time("scan.......\n",s5,s6); 
    //====================================================================================================
    //transferring data back to CPU
    uint *h_PKey=new uint[num_cells];
    uint *h_PLen=new uint[num_cells];
    uint *h_PPos=new uint[num_cells];    
    HANDLE_ERROR( cudaMemcpy( h_points, dptr_points, num_points * sizeof(point2d), cudaMemcpyDeviceToHost) ); 
    HANDLE_ERROR( cudaMemcpy( h_cellids, dptr_cellids, num_points * sizeof(uint), cudaMemcpyDeviceToHost) ); 
    HANDLE_ERROR( cudaMemcpy( h_PKey, dptr_PKey, num_cells * sizeof(uint), cudaMemcpyDeviceToHost) ); 
    HANDLE_ERROR( cudaMemcpy( h_PLen, dptr_PLen, num_cells * sizeof(uint), cudaMemcpyDeviceToHost) ); 
    HANDLE_ERROR( cudaMemcpy( h_PPos, dptr_PPos, num_cells * sizeof(uint), cudaMemcpyDeviceToHost) ); 
    gettimeofday(&s7, NULL);
    calc_time("transferring back to CPU.......\n",s6,s7);     
    
    //you would have to override the output opertor of point2d to output points to std::cout
    //thrust::copy(h_points, h_points+num_cells, std::ostream_iterator<point2d>(std::cout, " "));    
    
    //alternatively, you can access h_points array and print out x/y
    int point_out=(num_points>50)?50:num_points;
    for(int i=0;i<point_out;i++)
    {
     	point2d p=h_points[i];
     	printf("(%d,%d)",p.x,p.y);
     }
    printf("\n");
     
    cout<<"cell identifiers:";
    thrust::copy(h_cellids, h_cellids+point_out, std::ostream_iterator<uint>(std::cout, " "));       
    cout<<endl;
    int cell_out=(num_cells>20)?20:num_cells;
    cout<<"unique cell identifiers:";
    thrust::copy(h_PKey, h_PKey+cell_out, std::ostream_iterator<uint>(std::cout, " "));       
    cout<<endl;
    cout<<"number of points in cells:";
    thrust::copy(h_PLen, h_PLen+cell_out, std::ostream_iterator<uint>(std::cout, " "));       
    cout<<endl;
    cout<<"starting point position in cells:";
    thrust::copy(h_PPos, h_PPos+cell_out, std::ostream_iterator<uint>(std::cout, " "));       
    cout<<endl;
     
     //clean up
    cudaFree(dptr_points);
    cudaFree(dptr_cellids);
    cudaFree(dptr_PKey);
    cudaFree(dptr_PLen);
    cudaFree(dptr_PPos);
    delete[] h_points;
    delete[] h_cellids;
    delete[] h_PKey;
    delete[] h_PLen;
    delete[] h_PPos;
}
