
#include <iostream>
#include "cuda.h"

using Real = double;

//Test wrapper to run a function multiple times
template<typename PerfFunc>
float kernel_timer_wrapper(const int n_burn, const int n_perf, PerfFunc perf_func){

  //Initialize the timer and test
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for( int i_run = 0; i_run < n_burn + n_perf; i_run++){

    if(i_run == n_burn){
      //Burn in time is over, start timing
      cudaEventRecord(start);
    }

    //Run the function timing performance
    perf_func();
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  return milliseconds/1000.;
}

//////////////////////////////////////////////////////////////////////////////
//Parallelization strategy:
// Each block works on one variable for one face - when working with cube meshblocks, each
// block has the same number of points. The x direction, however, will not be
// coallesced
// 
// A block on face A  will have nghost*nxB*nxC points to load to a face
// Then two edges of size nghost*nghost*nxB and nghost*nghost*nxC respectively
// Then four edges of size  nghost**3 (but only for y and z faces)
//
// Will need nvar*nface blocks

//Should switch to one slice per block? So have nvar*nface*nghost blocks?
//////////////////////////////////////////////////////////////////////////////
__global__ void k_minbranch_scratch(Real* array4d_in, 
    Real* face_buf, Real* edge_buf, Real* vert_buf,
    const int nvar,
    const int nx1, const int nx2, const int nx3, const int nghost,
    const int int_nx1, const int int_nx2, const int int_nx3,
    const int var_face_buf_n, const int var_edge_buf_n
    ){

  //Setup face buffer in  Shared memory 
  extern __shared__ Real s_face_buf[];

  //Get index of face and variable to work on
  const int face = blockIdx.x;
  const int var = blockIdx.y;

  //Get direction (xyz) of face being worked on
  const int dir = face/3;
  //const int sign  = 2*(face%2)-1;

  const int thread_idx = threadIdx.x;
  const int thread_n = blockDim.x;

  //Compute offset from beginning of face buffer for this block/face
  const int face_buf_offset = var*var_face_buf_n + nghost*( 
      +( (2*(face>1) + (face==1))*( int_nx2*int_nx3) ) // +/- x faces
      +( (2*(face>3) + (face==3))*( int_nx3*int_nx1) ) // +/- y faces
      +( (             (face==5))*( int_nx1*int_nx2) ));// +/- z faces

  //Compute the start indices and dimensions of the face
  const int face_is = nghost + (face==1)*(int_nx1 - nghost);
  const int face_nx1= nghost*(dir==0) + (int_nx1)*(dir!=0);
  const int face_js = nghost + (face==3)*(int_nx2 - nghost);
  const int face_nx2= nghost*(dir==1) + (int_nx2)*(dir!=1);
  const int face_ks = nghost + (face==5)*(int_nx3 - nghost);
  const int face_nx3= nghost*(dir==2) + (int_nx3)*(dir!=2);

  //Comptue the size of the face slab for this block
  //const int gface_n = nghost*( ( dir==0 ? 1 : int_nx1)
  //                              *( dir==0 ? 1 : int_nx2)
  //                              *( dir==0 ? 1 : int_nx3));
  const int face_n = face_nx1*face_nx2*face_nx3;
  
  //Load in data for this face into shared memory
  for( int face_idx = thread_idx; face_idx < face_n; face_idx += thread_n){

    //Compute indices within the meshblock
    const int mb_k =  face_idx/( face_nx1*face_nx2 ) + face_ks;
    const int mb_j = (face_idx%( face_nx1*face_nx2 ))/face_nx1 + face_js;
    const int mb_i =  face_idx%face_nx1 + face_is;

    const int array_idx = mb_i + nx1*( mb_j + nx2*( mb_k + nx3*var));

    //Load the point into shared memory
    s_face_buf[face_idx] = array4d_in[array_idx];
    //Save it into the face buffer
    face_buf[face_idx+face_buf_offset] = s_face_buf[face_idx];
  }
  __syncthreads();



  //Compute starting indices and dimensions within face buffer of first edge
  const int edge1_is = (int_nx1-nghost)*(face==4);
  const int edge1_nx1 = int_nx1*(dir==1) + nghost*(dir!=1);
  const int edge1_js = (int_nx2-nghost)*(face==0);
  const int edge1_nx2 = int_nx2*(dir==2) + nghost*(dir!=2);
  const int edge1_ks = (int_nx3-nghost)*(face==2);
  const int edge1_nx3 = int_nx3*(dir==0) + nghost*(dir!=0);

  //Compute starting indices and dimensions within face buffer of second edge
  const int edge2_is = (int_nx1-nghost)*(face==3);
  const int edge2_nx1 = int_nx1*(dir==2) + nghost*(dir!=2);
  const int edge2_js = (int_nx2-nghost)*(face==5);
  const int edge2_nx2 = int_nx2*(dir==0) + nghost*(dir!=0);
  const int edge2_ks = (int_nx3-nghost)*(face==1);
  const int edge2_nx3 = int_nx3*(dir==1) + nghost*(dir!=1);

  //Compute total number of edge points handled by this block
  //const int edge1_n = nghost*nghost*(
  //     (dir==0)*int_nx3  // (x,y) edge
  //    +(dir==1)*int_nx1  // (y,z) edge
  //    +(dir==2)*int_nx1);// (z,x) edge
  const int edge1_n = edge1_nx1*edge1_nx2*edge1_nx3;
  //const int edge2_n = nghost*nghost*(
  //     (dir==0)*int_nx2   // (x,z) edge
  //    +(dir==1)*int_nx3   // (y,z) edge
  //    +(dir==2)*int_nx2 );// (z,y) edge
  const int edge2_n = edge2_nx1*edge2_nx2*edge2_nx3;
  const int edge_n = edge1_n + edge2_n;

  //Compute an offset within the edge buffer
  const int edge_buf_offset = var*var_edge_buf_n + nghost*nghost*(
      +( (2*(face>1) + (face==1))*( int_nx2 + int_nx3) )  // +/- x faces - y and z pencil
      +( (2*(face>3) + (face==3))*( int_nx3 + int_nx1) )  // +/- y faces - z and x pencil
      +( (             (face==5))*( int_nx1 + int_nx2) ));// +/- z faces - x and y pencil

  for( int edge_idx = thread_idx; edge_idx < edge_n; edge_idx += thread_n){
    const bool edge_flag = edge_idx > edge1_n;
    const int face_k =  (!edge_flag)*(edge_idx/( edge1_nx1*edge1_nx2 ) + edge1_ks)
                      + ( edge_flag)*(edge_idx/( edge2_nx1*edge2_nx2 ) + edge2_ks);
    const int face_j =  (!edge_flag)*((edge_idx%( edge1_nx1*edge1_nx2)/edge1_nx1 ) + edge1_js)
                      + ( edge_flag)*((edge_idx%( edge2_nx1*edge2_nx2)/edge2_nx1 ) + edge2_js);
    const int face_i =  (!edge_flag)*(edge_idx%edge1_nx1 + edge1_is)
                      + ( edge_flag)*(edge_idx%edge2_nx1 + edge2_is);

    edge_buf[edge_idx+edge_buf_offset] = s_face_buf[ face_i + face_nx1*(face_j + face_nx2*face_k) ];
  }

  if( dir != 0 ){
    //If not x-direction, work on the vertex buffer

    //Compute starting indices and dimensions within face buffer both loaded vertices
    const int vert_js = (face==5)*(int_nx2-nghost);
    const int vert_ks = (face==2)*(int_nx3-nghost);

    //Size of one vertex in buffer
    const int vert1_n = nghost*nghost*nghost;

    //Compute an offset within the vertex buffer
    //8 vertices for each variable, 2 for each previous face within this var
    const int vert_buf_offset = vert1_n*( 8*var + 2*(face-2) );

    for( int vert_idx = thread_idx; vert_idx < 2*vert1_n; vert_idx += thread_n){
      const int vert_is = (int_nx1-nghost)*(thread_idx < vert1_n);

      const int face_k =  vert_idx/( nghost*nghost ) + vert_ks;
      const int face_j = (vert_idx%( nghost*nghost))/nghost + vert_js;
      const int face_i =  vert_idx%nghost + vert_is;

      vert_buf[vert_idx+vert_buf_offset] = s_face_buf[ face_i + face_nx1*(face_j + face_nx2*face_k) ];
    }
  }

}



int main(int argc, char* argv[]) {
  //Load in parameters ( variable dimensions, number of variables)
  std::size_t pos;
  const int nx1 = std::stoi(argv[1],&pos);
  const int nx2 = std::stoi(argv[2],&pos);
  const int nx3 = std::stoi(argv[3],&pos);
  const int nvar = std::stoi(argv[4],&pos);
  const int nghost = std::stoi(argv[5],&pos);
  const int nrun = std::stoi(argv[6],&pos);

  //Total meshblock size
  const int nmb = nx1*nx2*nx3*nvar;

  //Lengths along interior edge
  const int int_nx1 = nx1 - 2*nghost;
  const int int_nx2 = nx2 - 2*nghost;
  const int int_nx3 = nx3 - 2*nghost;

  //Face buffers: 6 slabs of nghost thickness
  const int var_face_buf_n = 2*nghost*( int_nx2*int_nx3   //(+/-x faces)
                                          +int_nx3*int_nx1   //(+/-y faces)
                                          +int_nx1*int_nx2 );//(+/-z faces)
  const int  total_face_buf_n = nvar*var_face_buf_n;

  //Edge buffers: 12 pencils of nghost*nghost size
  const int var_edge_buf_n = 3*nghost*nghost*( int_nx1 + int_nx2 + int_nx3);
  const int total_edge_buf_n = nvar*var_edge_buf_n;

  //Vertex buffers: 8 cubes of nghost*nghost*nghost size
  const int var_vert_buf_n = 8*(nghost*nghost*nghost);
  const int total_vert_buf_n = nvar*var_vert_buf_n;

  //Total buffer size
  const int total_buf = total_face_buf_n + total_edge_buf_n + total_vert_buf_n;

  //Setup a 4d data array and 1d buffer with faces, edges, verts
  Real* d_array4d_in;
  cudaMalloc(&d_array4d_in, sizeof(Real)*nmb);
  Real* d_buf;
  cudaMalloc(&d_buf, sizeof(Real)*(total_buf) );

  //Make some more convenient device pointers
  Real* d_face_buf = d_buf;
  Real* d_edge_buf = d_face_buf + total_face_buf_n;
  Real* d_vert_buf = d_edge_buf + total_edge_buf_n;


  //Define parallelization
  const int threads_per_block = 64;
  //One block in grid.x for each face (+-xyz)
  //One block in grid.y for each variable
  const dim3 cuda_grid(6,nvar,1);
  const dim3 cuda_block(threads_per_block,1,1);


  //Allocate shared memory to accomodate the largest face buffer
  const  size_t shared_memory_size = sizeof(Real) * nghost * 
    max(  max(int_nx1*int_nx2, int_nx2*int_nx3), int_nx3*int_nx1);

  float time_minbranch_scratch = kernel_timer_wrapper( nrun, nrun,
    [&] () {

      k_minbranch_scratch<<< cuda_grid, cuda_block, shared_memory_size >>> 
        (d_array4d_in, 
         d_face_buf, d_edge_buf, d_face_buf,
         nvar,
         nx1, nx2, nx3, nghost,
         int_nx1, int_nx2, int_nx3,
         var_face_buf_n, var_edge_buf_n
         );

  });


  double cell_cycles_per_second_minbranch_scratch = static_cast<double>(nmb)*static_cast<double>(nrun)/time_minbranch_scratch; 
  std::cout<< nvar << " " << nmb << " " << nrun << " " 
           << time_minbranch_scratch << " " << cell_cycles_per_second_minbranch_scratch << " " << 
           std::endl;



  cudaFree(d_array4d_in);
  cudaFree(d_buf);

}
