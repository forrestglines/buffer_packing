#include <Kokkos_Core.hpp>

#include <iostream>
#include <vector>
#include <assert.h>

using Real = double;
using View1D =  Kokkos::View<Real*   , Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace >;
using View4D =  Kokkos::View<Real****, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace >;

using HView1D =  View1D::HostMirror;
using HView4D =  View4D::HostMirror;

using HIntView1D =  Kokkos::View<int*   , Kokkos::HostSpace >;
using HIntView2D =  Kokkos::View<int**  , Kokkos::HostSpace >;

//Test wrapper to run a function multiple times
template<typename PerfFunc>
double kernel_timer_wrapper(const int n_burn, const int n_perf, PerfFunc perf_func){

  //Initialize the timer and test
  Kokkos::Timer timer;

  for( int i_run = 0; i_run < n_burn + n_perf; i_run++){

    if(i_run == n_burn){
      //Burn in time is over, start timing
      Kokkos::fence();
      timer.reset();
    }

    //Run the function timing performance
    perf_func();
  }

  //Time it
  Kokkos::fence();
  double perf_time = timer.seconds();

  return perf_time;

}


//Names and orders of all the buffers
//Ordering can be changed for optimization
enum BufferName {
//The 6 faces
  XM_FACE,
  XP_FACE,
  YM_FACE,
  YP_FACE,
  ZM_FACE,
  ZP_FACE,
//The 12 edges
  X_YM_ZM_EDGE,
  X_YM_ZP_EDGE,
  X_YP_ZM_EDGE,
  X_YP_ZP_EDGE,

  Y_XM_ZM_EDGE,
  Y_XM_ZP_EDGE,
  Y_XP_ZM_EDGE,
  Y_XP_ZP_EDGE,

  Z_XM_YM_EDGE,
  Z_XM_YP_EDGE,
  Z_XP_YM_EDGE,
  Z_XP_YP_EDGE,

//The 8 verts
  XM_YM_ZM_EDGE,
  XM_YM_ZP_EDGE,
  XM_YP_ZM_EDGE,
  XM_YP_ZP_EDGE,
  XP_YM_ZM_EDGE,
  XP_YM_ZP_EDGE,
  XP_YP_ZM_EDGE,
  XP_YP_ZP_EDGE
};
const int N_BUFFERS=26;

//////////////////////////////////////////////////////////////////////////////
//Compute offsets into buffer
//
//Buffer Organization;
//  The grouping of buffers, from slowest moving to fastest moving is:
//  (variables) . (faces/edges/verts) . (z) . (y) . (x)
//
//  This function computes the offsets of the buffers within one variables, so
//  that these offsets can reused across variables
//////////////////////////////////////////////////////////////////////////////
void compute_buf_offsets( HIntView1D buf_offsets,
    const int nvar, const int nx1, const int nx2, const int nx3, const int nghost,
    const int int_nx1, const int int_nx2, const int int_nx3,
    const int var_face_buf_n, const int var_edge_buf_n, const int var_vert_buf_n,
    const int var_buf_n
    ) {
  int offset_idx = 0;
  int buf_offset = 0;

  // +/- x faces
  for( int i = 0; i < 2; i++){
    buf_offsets(offset_idx++) = buf_offset;
    buf_offset += nghost*int_nx2*int_nx3*nvar;
  }

  // +/- y faces
  for( int i = 0; i < 2; i++){
    buf_offsets(offset_idx++) = buf_offset;
    buf_offset += nghost*int_nx1*int_nx3*nvar;
  }

  // +/- z faces
  for( int i = 0; i < 2; i++){
    buf_offsets(offset_idx++) = buf_offset;
    buf_offset += nghost*int_nx1*int_nx2*nvar;
  }

  assert( buf_offset == (var_face_buf_n)*nvar);

  // x edges
  for( int i = 0; i < 4; i++){
    buf_offsets(offset_idx++) = buf_offset;
    buf_offset += nghost*nghost*int_nx1*nvar;
  }

  // y edges
  for( int i = 0; i < 4; i++){
    buf_offsets(offset_idx++) = buf_offset;
    buf_offset += nghost*nghost*int_nx2*nvar;
  }

  // z edges
  for( int i = 0; i < 4; i++){
    buf_offsets(offset_idx++) = buf_offset;
    buf_offset += nghost*nghost*int_nx3*nvar;
  }

  assert( buf_offset == (var_face_buf_n + var_edge_buf_n)*nvar);

  // the verts
  for( int i = 0; i < 8; i++){
    buf_offsets(offset_idx++) = buf_offset;
    buf_offset += nghost*nghost*nghost;
  }

  assert( buf_offset == (var_face_buf_n + var_edge_buf_n + var_vert_buf_n)*nvar);
  assert( buf_offset == (var_buf_n)*nvar);
}

void compute_buf_boxes( const HIntView2D& buf_boxes,
    const int& nx1, const int& nx2, const int& nx3, const int& nghost,
    const int& int_nx1, const int& int_nx2, const int& int_nx3,
    const int& var_buf_n
    ) {
  int buf_idx = 0;

  int total_var_buf_n = 0;

  // Faces
  for( int face_dim = 0; face_dim < 3; face_dim++){
    for( int face_side = 0; face_side < 2; face_side++){
      //Start indices of face
      const int face_is = (face_dim == 0 && face_side ) ? int_nx1 : nghost;
      const int face_js = (face_dim == 1 && face_side ) ? int_nx2 : nghost;
      const int face_ks = (face_dim == 2 && face_side ) ? int_nx3 : nghost;

      //Dimensions of the face
      const int face_nx1 = (face_dim == 0) ? nghost : int_nx1;
      const int face_nx2 = (face_dim == 1) ? nghost : int_nx2;
      const int face_nx3 = (face_dim == 2) ? nghost : int_nx3;
      
      //Store this buffer box
      buf_boxes( buf_idx ,0 ) = face_is;
      buf_boxes( buf_idx ,1 ) = face_nx1;
      buf_boxes( buf_idx ,2 ) = face_js;
      buf_boxes( buf_idx ,3 ) = face_nx2;
      buf_boxes( buf_idx ,4 ) = face_ks;
      buf_boxes( buf_idx ,5 ) = face_nx3;
      buf_idx++;

      total_var_buf_n += face_nx1*face_nx2*face_nx3;
    }
  }

  // edges
  for( int edge_dim = 0; edge_dim < 3; edge_dim++){
    const int edge1_dim = (edge_dim+1)%3;
    const int edge2_dim = (edge_dim+2)%3;
    for( int edge1_side = 0; edge1_side < 2; edge1_side++){
      for( int edge2_side = 0; edge2_side < 2; edge2_side++){
        //Start indices of edge
        const int edge_is = (edge_dim == 0 || (edge1_dim == 0 && !edge1_side) || (edge2_dim == 0 && !edge2_side ) ) ? nghost : int_nx1;
        const int edge_js = (edge_dim == 1 || (edge1_dim == 1 && !edge1_side) || (edge2_dim == 1 && !edge2_side ) ) ? nghost : int_nx2;
        const int edge_ks = (edge_dim == 2 || (edge1_dim == 2 && !edge1_side) || (edge2_dim == 2 && !edge2_side ) ) ? nghost : int_nx3;

        //Dimensions of the edge
        const int edge_nx1 = (edge_dim == 0) ? int_nx1 : nghost;
        const int edge_nx2 = (edge_dim == 1) ? int_nx2 : nghost;
        const int edge_nx3 = (edge_dim == 2) ? int_nx3 : nghost;

        
        //Store this buffer box
        buf_boxes( buf_idx ,0 ) = edge_is;
        buf_boxes( buf_idx ,1 ) = edge_nx1;
        buf_boxes( buf_idx ,2 ) = edge_js;
        buf_boxes( buf_idx ,3 ) = edge_nx2;
        buf_boxes( buf_idx ,4 ) = edge_ks;
        buf_boxes( buf_idx ,5 ) = edge_nx3;
        buf_idx++;

        total_var_buf_n += edge_nx1*edge_nx2*edge_nx3;
      }
    }
  }

  for( int side1 = 0; side1 < 2; side1++){
    for( int side2 = 0; side2 < 2; side2++){
      for( int side3 = 0; side3 < 2; side3++){
        //Start indices of vert
        const int vert_is = side1 ? int_nx1 : nghost;
        const int vert_js = side2 ? int_nx2 : nghost;
        const int vert_ks = side3 ? int_nx3 : nghost;

        //Dimensions of the vert
        const int vert_nx1 = nghost;
        const int vert_nx2 = nghost;
        const int vert_nx3 = nghost;

        
        //Store this buffer box
        buf_boxes( buf_idx ,0 ) = vert_is;
        buf_boxes( buf_idx ,1 ) = vert_nx1;
        buf_boxes( buf_idx ,2 ) = vert_js;
        buf_boxes( buf_idx ,3 ) = vert_nx2;
        buf_boxes( buf_idx ,4 ) = vert_ks;
        buf_boxes( buf_idx ,5 ) = vert_nx3;
        buf_idx++;

        total_var_buf_n += vert_nx1*vert_nx2*vert_nx3;
      }
    }
  }

  assert( buf_idx == N_BUFFERS);
  assert( total_var_buf_n == var_buf_n);
}

void cpp_buffer_packing( const HView4D& in, const HView1D& buf,
    const int& nvar, const int var_buf_n,
    const HIntView1D& buf_offsets, const HIntView2D& buf_boxes
    ){


  for( int l = 0; l < nvar ; l++){
    for( int buf_idx =0; buf_idx < N_BUFFERS; buf_idx++){
      //Get the start indices and length of this buffer
      const int buf_is  = buf_boxes(buf_idx, 0);
      const int buf_nx1 = buf_boxes(buf_idx, 1);
      const int buf_js  = buf_boxes(buf_idx, 2);
      const int buf_nx2 = buf_boxes(buf_idx, 3);
      const int buf_ks  = buf_boxes(buf_idx, 4);
      const int buf_nx3 = buf_boxes(buf_idx, 5);

      for( int buf_k = 0; buf_k < buf_nx3; buf_k++){
        for( int buf_j = 0; buf_j < buf_nx2; buf_j++){
          for( int buf_i = 0; buf_i < buf_nx1; buf_i++){
            const int i = buf_i + buf_is;
            const int j = buf_j + buf_js;
            const int k = buf_k + buf_ks;

            const int this_buf_idx = buf_i + buf_nx1*( buf_j + buf_nx2*buf_k);
            const int all_buf_idx = buf_offsets(buf_idx) + buf_nx1*buf_nx2*buf_nx3*l + this_buf_idx;

            buf( all_buf_idx ) = in(l,k,j,i);
          }
        }
      }
    
    }//End this buffer
  }//End variables
}

bool compare_buffers( const HView1D& buf_gold, const HView1D& buf_comp, const HView4D& in,
    const int& nvar, const int var_buf_n,
    const HIntView1D& buf_offsets, const HIntView2D& buf_boxes
    ){


  bool all_matching = true;
  for( int l = 0; l < nvar ; l++){
    for( int buf_idx =0; buf_idx < N_BUFFERS; buf_idx++){
      //Get the start indices and length of this buffer
      const int buf_is  = buf_boxes(buf_idx, 0);
      const int buf_nx1 = buf_boxes(buf_idx, 1);
      const int buf_js  = buf_boxes(buf_idx, 2);
      const int buf_nx2 = buf_boxes(buf_idx, 3);
      const int buf_ks  = buf_boxes(buf_idx, 4);
      const int buf_nx3 = buf_boxes(buf_idx, 5);

      for( int buf_k = 0; buf_k < buf_nx3; buf_k++){
        for( int buf_j = 0; buf_j < buf_nx2; buf_j++){
          for( int buf_i = 0; buf_i < buf_nx1; buf_i++){

            const int this_buf_idx = buf_i + buf_nx1*( buf_j + buf_nx2*buf_k);

            const int all_buf_idx = buf_offsets(buf_idx) + buf_nx1*buf_nx2*buf_nx3*l + this_buf_idx;

            bool matches = (buf_gold( all_buf_idx ) 
                         == buf_comp( all_buf_idx ));

            if( !matches ){
              const int i = buf_i + buf_is;
              const int j = buf_j + buf_js;
              const int k = buf_k + buf_ks;
              assert( in(l,k,j,i) == buf_gold( all_buf_idx) );

              std::cout<< "Fault in buf: " << buf_idx
                       << " at buf (" << buf_i << "," << buf_j << "," << buf_k <<" ). "
                       << " at mb  (" << i << "," << j << "," << k <<" ). "
                       << " buf_gold("<< all_buf_idx << ")= "
                       << buf_gold( all_buf_idx )
                       << " buf_comp("<< all_buf_idx << ")= "
                       << buf_comp( all_buf_idx )
                       << std::endl;
            }

            assert(matches);

            all_matching &= matches;
          }
        }
      }
    
    }//End this buffer
  }//End variables

  assert(all_matching);
  return all_matching;
}


template<typename Functor>
bool check_against_cpp( Functor functor,
    const View4D& in, const View1D& buf,
    const int& nvar,
    const int& nx1, const int& nx2, const int& nx3, const int& nghost,
    const int& int_nx1, const int& int_nx2, const int& int_nx3,
    const int& var_face_buf_n, const int& var_edge_buf_n, const int& var_vert_buf_n,
    const int& var_buf_n, const int& total_buf
    ){

  //Create host copy of in
  HView4D h_in = Kokkos::create_mirror_view(in);
  Kokkos::deep_copy(h_in,in);

  //Create host buffer
  HView1D h_buf("h_buf",total_buf);

  //Create buffer offsets and boxes views
  HIntView1D h_buf_offsets("h_buf_offsets",N_BUFFERS);
  compute_buf_offsets( h_buf_offsets,
      nvar, nx1, nx2, nx3, nghost,
      int_nx1, int_nx2, int_nx3,
      var_face_buf_n, var_edge_buf_n, var_vert_buf_n,
      var_buf_n);
  HIntView2D h_buf_boxes("h_buf_boxes",N_BUFFERS,6);
  compute_buf_boxes( h_buf_boxes,
      nx1, nx2, nx3, nghost,
      int_nx1, int_nx2, int_nx3,
      var_buf_n);

  //Do buffer packing on host
  cpp_buffer_packing( h_in, h_buf,
      nvar, var_buf_n,
      h_buf_offsets, h_buf_boxes);

  //Do buffer packing on device
  Kokkos::fence();
  functor();
  Kokkos::fence();

  //Create host copy of buf
  HView1D h_buf_copy = Kokkos::create_mirror_view(buf);
  Kokkos::deep_copy(h_buf_copy,buf);

  //Compare two buffers
  compare_buffers( h_buf, h_buf_copy, h_in,
      nvar, var_buf_n,
      h_buf_offsets, h_buf_boxes);

  return true;
}


//Save "data" at "l,k,j,i" to "buf" in the "face_dim,face_side" face buffer
KOKKOS_INLINE_FUNCTION void save_to_face_buf( const View4D& in, const View1D& buf,
    const Real& data,
    const int& l, const int& k, const int& j, const int& i,
    const int& face_dim, const bool& face_side,
    const int& nvar,
    const int& nx1, const int& nx2, const int& nx3, const int& nghost,
    const int& int_nx1, const int& int_nx2, const int& int_nx3
    ){
  //Start indices of face
  const int face_is = (face_dim == 0 && face_side ) ? int_nx1 : nghost;
  const int face_js = (face_dim == 1 && face_side ) ? int_nx2 : nghost;
  const int face_ks = (face_dim == 2 && face_side ) ? int_nx3 : nghost;

  //Dimensions of the face
  const int face_nx1 = (face_dim == 0) ? nghost : int_nx1;
  const int face_nx2 = (face_dim == 1) ? nghost : int_nx2;
  const int face_nx3 = (face_dim == 2) ? nghost : int_nx3;

  //Index of k,j,i within the face
  const int face_i = i - face_is;
  const int face_j = j - face_js;
  const int face_k = k - face_ks;

  //Index within face buffer
  const int face_buf_idx  =  face_i + face_nx1*( face_j + face_nx2*face_k);

  //Offset in buf
  const int buf_offset = nvar*2*nghost*( (face_dim > 0)*int_nx2*int_nx3 + (face_dim > 1)*int_nx1*int_nx3) //Offest from previous faces
                       + face_nx1*face_nx2*face_nx3*l //Offset from previous variables
                       + face_side*face_nx1*face_nx2*face_nx3; //Add M face offset, if applicable

  const int buf_idx = buf_offset + face_buf_idx;
  //Save data to buf
  buf( buf_idx ) = data;
}

//Save "data" at "l,k,j,i" to "buf" in the "edge_dim,edge1_side,edge2_side" edge buffer
KOKKOS_INLINE_FUNCTION void save_to_edge_buf( const View4D& in, const View1D& buf,
    const Real& data,
    const int& l, const int& k, const int& j, const int& i,
    const int& edge_dim, const int& edge1_dim, const bool& edge1_side, const int& edge2_dim, const bool& edge2_side,
    const int& nvar,
    const int& nx1, const int& nx2, const int& nx3, const int& nghost,
    const int& int_nx1, const int& int_nx2, const int& int_nx3,
    const int& total_face_buf_n
    ){
  //Start indices of edge
  const int edge_is = (edge_dim == 0 || (edge1_dim == 0 && !edge1_side) || (edge2_dim == 0 && !edge2_side ) ) ? nghost : int_nx1;
  const int edge_js = (edge_dim == 1 || (edge1_dim == 1 && !edge1_side) || (edge2_dim == 1 && !edge2_side ) ) ? nghost : int_nx2;
  const int edge_ks = (edge_dim == 2 || (edge1_dim == 2 && !edge1_side) || (edge2_dim == 2 && !edge2_side ) ) ? nghost : int_nx3;

  //Dimensions of the edge
  const int edge_nx1 = (edge_dim == 0) ? int_nx1 : nghost;
  const int edge_nx2 = (edge_dim == 1) ? int_nx2 : nghost;
  const int edge_nx3 = (edge_dim == 2) ? int_nx3 : nghost;

  //Index of k,j,i within the edge
  const int edge_i = i - edge_is;
  const int edge_j = j - edge_js;
  const int edge_k = k - edge_ks;

  //Index within edge buffer
  const int edge_buf_idx  =  edge_i + edge_nx1*( edge_j + edge_nx2*edge_k);


  //Offset in buf
  const int buf_offset = + total_face_buf_n //Add offset from faces
                         + nvar*4*nghost*nghost*( (edge_dim > 0)*int_nx1 + (edge_dim > 1)*int_nx2) //Offest from previous edges in other dims
                         + edge_nx1*edge_nx2*edge_nx3*l //Offset from previous variables
                         + (2*edge1_side + edge2_side)*edge_nx1*edge_nx2*edge_nx3; //Add offset from previous edge in this dim

  const int buf_idx = buf_offset + edge_buf_idx;
  //Save data to buf
  buf( buf_idx ) = data;
}

//Save "data" at "l,k,j,i" to "buf" in the "side1, side2, side3" vert buffer
KOKKOS_INLINE_FUNCTION void save_to_vert_buf( const View4D& in, const View1D& buf,
    const Real& data,
    const int& l, const int& k, const int& j, const int& i,
    const bool& side1, const bool& side2, const bool& side3,
    const int& nvar,
    const int& nx1, const int& nx2, const int& nx3, const int& nghost,
    const int& int_nx1, const int& int_nx2, const int& int_nx3,
    const int& total_face_buf_n, const int& total_edge_buf_n
    ){
  //Start indices of vert
  const int vert_is = side1 ? int_nx1 : nghost;
  const int vert_js = side2 ? int_nx2 : nghost;
  const int vert_ks = side3 ? int_nx3 : nghost;

  //Dimensions of the vert
  const int vert_nx1 = nghost;
  const int vert_nx2 = nghost;
  const int vert_nx3 = nghost;

  //Index of k,j,i within the vert
  const int vert_i = i - vert_is;
  const int vert_j = j - vert_js;
  const int vert_k = k - vert_ks;

  //Index within vert buffer
  const int vert_buf_idx  =  vert_i + vert_nx1*( vert_j + vert_nx2*vert_k);

  //Offset in buf
  const int buf_offset = total_face_buf_n //Add offset from faces
                       + total_edge_buf_n //Add offset from edges
                       + vert_nx1*vert_nx2*vert_nx3*l
                       + nghost*nghost*nghost*( 4*side1 + 2*side2 + side3 ); //Offest from previous verts

  const int buf_idx = buf_offset + vert_buf_idx;
  //Save data to buf
  buf( buf_idx ) = data;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
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
    const int var_face_buf_n = 2*nghost*( int_nx2*int_nx3   //(+/-x faces)   //size of all faces for one var
                                            +int_nx3*int_nx1   //(+/-y faces)
                                            +int_nx1*int_nx2 );//(+/-z faces)
    const int total_face_buf_n = nvar*var_face_buf_n; //Total size of all faces

    //Edge buffers: 12 pencils of nghost*nghost size
    const int var_edge_buf_n = 4*nghost*nghost*( int_nx1 + int_nx2 + int_nx3);//Size of all edges for one var
    const int total_edge_buf_n = nvar*var_edge_buf_n; //Total size of all edges

    //Vertex buffers: 8 cubes of nghost*nghost*nghost size
    const int var_vert_buf_n = 8*(nghost*nghost*nghost); //Size of all vertices for one var
    const int total_vert_buf_n = nvar*var_vert_buf_n; //Tota size of all vertices

    //Buffer size for one variable
    const int var_buf_n = var_face_buf_n + var_edge_buf_n + var_vert_buf_n;

    //Total buffer size
    const int total_buf = nvar*var_buf_n;
    assert( total_buf == total_face_buf_n + total_edge_buf_n + total_vert_buf_n);

    //Setup the input 4D view
    View4D in("in",  nvar,nx3,nx2,nx1);

    Kokkos::parallel_for( "Setup Loop", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nvar,nx3,nx2,nx1}),
      KOKKOS_LAMBDA (const int &l, const int &k, const int& j, const int& i){
        const int unique_idx = i + nx1*(j + nx2*(k + nx3*l));
        in(l,k,j,i) = static_cast<double>(unique_idx);
    }); 

    //Setup face,edge,vertex buffers
    View1D buf("buf",total_buf);

    //Simple loop

    //Number of points that are only in one of the x faces
    const int x_slab_points = nghost * (int_nx2 - 2*nghost) * (int_nx3 - 2*nghost);
    //Number of points that are in one of the y faces but not z faces
    const int y_slab_points = nghost * int_nx1 * (int_nx3 - 2*nghost);
    //Number of points that are in one the z faces
    const int z_slab_points = nghost * int_nx1 * int_nx2;

    auto simple_kernel = [&] () {
      Kokkos::parallel_for( "Simple Loop", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nvar, 2*(x_slab_points+y_slab_points+z_slab_points)}),
        KOKKOS_LAMBDA (const int& l, const int& idx){
          //Indices into the mesh block
          int i,j,k;

          //Load data from in
          Real data;
          //Determine which group of points idx belongs to
          if( idx < 2*x_slab_points){
            //idx is in a x slab (no edges)
            const int slab_idx = idx%x_slab_points;//idx inside the slab
            const int slab_side = idx >= x_slab_points;//Which side to work on

            //Start and dimensions of slab
            const int slab_is = slab_side ? nghost : int_nx1;//Is this XM or XP?
            const int slab_nx1 = nghost;
            const int slab_js = 2*nghost; //Exclude the edges, take care of by other slabs
            const int slab_nx2 = int_nx2 - nghost*2;
            const int slab_ks = 2*nghost;
            //const int slab_nx3 = int_nx3 - nghost*2;
            
            //Find the index within the grid
            k =  slab_idx/( slab_nx1*slab_nx2 ) + slab_ks;
            j = (slab_idx%( slab_nx1*slab_nx2 ))/slab_nx1 + slab_js;
            i =  slab_idx%slab_nx1 + slab_is;

            assert( i - slab_is + slab_nx1*( j - slab_js + slab_nx2 * (k - slab_ks)) == slab_idx );
            
            //Load the data point
            data = in(l,k,j,i);
          } else if ( idx < 2*( x_slab_points + y_slab_points) ){
            //idx is in a y slab (no z edges)
            const int slab_idx = (idx-2*x_slab_points) %y_slab_points;//idx inside the slab
            const int slab_side = (idx-2*x_slab_points) >= y_slab_points;//Which side to work on

            //Start and dimensions of slab
            const int slab_is = nghost;
            const int slab_nx1 = int_nx1;
            const int slab_js = slab_side ? nghost : int_nx2; //Is this YM or YP?
            const int slab_nx2 = nghost;
            const int slab_ks = 2*nghost; ///Exclude the edges
            //const int slab_nx3 = int_nx3 - nghost*2;
            
            //Find the index within the grid
            k =  slab_idx/( slab_nx1*slab_nx2 ) + slab_ks;
            j = (slab_idx%( slab_nx1*slab_nx2 ))/slab_nx1 + slab_js;
            i =  slab_idx%slab_nx1 + slab_is;
            
            assert( i - slab_is + slab_nx1*( j - slab_js + slab_nx2 * (k - slab_ks)) == slab_idx );

            //Load the data point
            data = in(l,k,j,i);
          } else {
            //idx is in a z slab (all edges)
            const int slab_idx = (idx-2*(x_slab_points+y_slab_points)) %z_slab_points;//idx inside the slab
            const int slab_side = (idx-2*(x_slab_points+y_slab_points)) >= z_slab_points;//Which side to work on

            //Start and dimensions of slab
            const int slab_is = nghost;
            const int slab_nx1 = int_nx1;
            const int slab_js = nghost;
            const int slab_nx2 = int_nx2;
            const int slab_ks = slab_side ? nghost : int_nx3; //Is this ZM or ZP?
            //const int slab_nx3 = nghost;
            
            //Find the index within the grid
            k =  slab_idx/( slab_nx1*slab_nx2 ) + slab_ks;
            j = (slab_idx%( slab_nx1*slab_nx2 ))/slab_nx1 + slab_js;
            i =  slab_idx%slab_nx1 + slab_is;

            assert( i - slab_is + slab_nx1*( j - slab_js + slab_nx2 * (k - slab_ks)) == slab_idx );

            //Load the data point
            data = in(l,k,j,i);
          }

          //Handy arrays for simplifying code
          const int idx_array[] = {i,j,k};
          const int int_nx_array[] = {int_nx1,int_nx2,int_nx3};

          //Save faces
          for(int dim = 0; dim < 3; dim++){
            //Determine if is a face
            const bool is_face = (idx_array[dim] < 2*nghost) || (idx_array[dim] >= int_nx_array[dim]) ;
            if( is_face){
              //Determine which face side
              const bool face_side = (idx_array[dim] >= int_nx_array[dim]);
              //Save to the face buffer
              save_to_face_buf( in, buf,
                  data,
                  l, k, j, i,
                  dim,face_side,
                  nvar,
                  nx1, nx2, nx3, nghost,
                  int_nx1, int_nx2, int_nx3);
            }
          }

          //Save edges
          for(int dim = 0; dim < 3; dim++){
            //Compute dimensions of other edge dimensions (the pencil width/height)
            const int edge1_dim = (dim+1)%3;
            const int edge2_dim = (dim+2)%3;

            //alias indices along other edge dimensions
            const int edge1_idx = idx_array[edge1_dim];
            const int edge2_idx = idx_array[edge2_dim];

            //alias int_nx of other edge dimensions
            const int edge1_int_nx = int_nx_array[edge1_dim];
            const int edge2_int_nx = int_nx_array[edge2_dim];

            //Determine if is an edge
            const bool is_edge = ( ( edge1_idx < 2*nghost || edge1_idx >= edge1_int_nx ) 
                                && ( edge2_idx < 2*nghost || edge2_idx >= edge2_int_nx ) );
            if( is_edge){
              //Determine which side of each other edge dimesion
              const bool edge1_side =  edge1_idx >= edge1_int_nx;
              const bool edge2_side =  edge2_idx >= edge2_int_nx;
              //Save to the edge buffer
              save_to_edge_buf( in, buf,
                  data,
                  l, k, j, i,
                  dim, edge1_dim, edge1_side, edge2_dim, edge2_side,
                  nvar,
                  nx1, nx2, nx3, nghost,
                  int_nx1, int_nx2, int_nx3,
                  total_face_buf_n);
            }

          }

          //Save verts
          const bool is_vert = ( ( i < 2*nghost || i >= int_nx1 ) 
                              && ( j < 2*nghost || j >= int_nx2 ) 
                              && ( k < 2*nghost || k >= int_nx3 ) );
          if( is_vert){
            //Determine which side of each vert dimesion
            const bool side1 = i >= int_nx1;
            const bool side2 = j >= int_nx2;
            const bool side3 = k >= int_nx3;
            //Save to the edge buffer
            save_to_vert_buf( in, buf,
                data,
                l, k, j, i,
                side1, side2, side3,
                nvar,
                nx1, nx2, nx3, nghost,
                int_nx1, int_nx2, int_nx3,
                total_face_buf_n, total_edge_buf_n);
          }
      });  //End lambda, parallel_for
    }; //End lambda

    check_against_cpp(simple_kernel,
        in, buf,
        nvar,
        nx1, nx2, nx3, nghost,
        int_nx1, int_nx2, int_nx3,
        var_face_buf_n, var_edge_buf_n, var_vert_buf_n,
        var_buf_n, total_buf);

    double time_simple = kernel_timer_wrapper( nrun, nrun, simple_kernel);

    double cell_cycles_per_second_simple = static_cast<double>(nmb)*static_cast<double>(nrun)/time_simple; 
    double time_per_kernel_simple = time_simple/static_cast<double>(nrun); 
    std::cout<< nvar << " " << nx1 << " " << nx2 << " " << nx3 << " " << " " << nmb << " " << nrun << " " 
             << time_simple << " " << time_per_kernel_simple << " " << cell_cycles_per_second_simple << " " << 
             std::endl;
  }
  Kokkos::finalize();
}
