#include <Kokkos_Core.hpp>

#include <iostream>
#include <vector>

using Real = double;
using View1D =  Kokkos::View<Real*   , Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace >;
using View4D =  Kokkos::View<Real****, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace >;

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


    //Lengths along interior edge
    const int int_nx1 = nx1 - 2*nghost;
    const int int_nx2 = nx2 - 2*nghost;
    const int int_nx3 = nx3 - 2*nghost;

    //Face buffers: 6 slabs of nghost thickness
    const int total_face_buf_size = 2*nghost*( int_nx2*int_nx3   //(+/-x faces)
                                              +int_nx3*int_nx1   //(+/-y faces)
                                              +int_nx1*int_nx2 );//(+/-z faces)
    //Edge buffers: 12 pencils of nghost*nghost size
    const int total_edge_buf_size = 3*nghost*nghost*( int_nx1 + int_nx2 + int_nx3);

    //Vertex buffers: 8 cubes of nghost*nghost*nghost size
    const int total_vert_buf_size = 8*(nghost*nghost*nghost)

    //Setup the input 4D view
    View4D view4d_in("view4D_in",  nvar,nx3,nx2,nx1);

    Kokkos::parallel_for( "Setup Loop", Kokkos::MDRangePolicy<Kokkos::Rank<4>>{nx3,nx2,nx1},
      KOKKOS_LAMBDA (const int &l, const int &k, const int& j, const int& i){
        const int unique_idx = i + nx1*(j + nx2*(k + nx3*l));
        view2d_in(l,k,j,i) = 2.*static_cast<double>(unique_idx);
    }); 

    //Setup face,edge,vertex buffers
    View1D face_buf("face_buf",total_face_buf_size);
    View1D edge_buf("edge_buf",total_edge_buf_size);
    View1D vert_buf("vert_buf",total_vert_buf_size);


    double time_minbranch_noscratch = kernel_timer_wrapper( n_run, n_run,
      [&] () {
          //TODO
    });

    double cell_cycles_per_second_view2d = static_cast<double>(n_grid)*static_cast<double>(n_run)/time_view2d; 
    double cell_cycles_per_second_view_of_view1d = static_cast<double>(n_grid)*static_cast<double>(n_run)/time_view_of_view1d; 
    std::cout<< n_var << " " << n_grid << " " << n_run  << " " << time_view2d << " " << time_view_of_view1d << " " 
             << cell_cycles_per_second_view2d << " " << cell_cycles_per_second_view_of_view1d << std::endl;
  }
  Kokkos::finalize();
}
