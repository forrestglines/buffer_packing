# buffer_packing
Performance testing repo for packing/unpacking ghostzones


## Checking out the repo
```

#Create a directory for the project, including builds and source
mkdir buffer_packing_project
cd buffer_packing_project

#Clone the repository
git clone https://github.com/forrestglines/buffer_packing

#Clone the submodules to get Kokkos into the source
cd buffer_packing
git submodule init
git submodule update
```

## Building on Summit (Volta70+Power9)

Loading the environment
```
#PWD=buffer_packing_project
SOURCE_DIR=buffer_packing
KOKKOS_DIR=$SOURCE_DIR/external/Kokkos

module swap xl gcc 
module load cuda
module load cmake
```

Building for CUDA debugging
```
#PWD=buffer_packing_project
mkdir -p  builds/cuda-Release-volta70
cd builds/cuda-Release-volta70

cmake \
    -D CMAKE_BUILD_TYPE=Debug \
    -D CMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
    -D CMAKE_CXX_FLAG="--Xptxas -G" \
    -D Kokkos_ARCH_VOLTA70=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
    -D Kokkos_ARCH_POWER9=ON \
    $SOURCE_DIR
make
```

Building for CUDA Release
```
#PWD=buffer_packing_project
mkdir -p  builds/cuda-Release-volta70
cd builds/cuda-Release-volta70

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
    -D Kokkos_ARCH_VOLTA70=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
    -D Kokkos_ARCH_POWER9=ON \
    $SOURCE_DIR
make
```

Building for GNU Debugging
```
#PWD=buffer_packing_project
mkdir -p  builds/gnu-Debug-serial
cd builds/gnu-Debug-serial
cmake \
    -D CMAKE_BUILD_TYPE=Debug \
    -D CMAKE_CXX_FLAG="-g" \
    $SOURCE_DIR
make
```

## Running
```
#PWD=buffer_packing_project/builds/cuda-Release-volta70
./tests/kokkos_buffer_packing/kokkos_buffer_packing $nx1 $nx2 $nx2 $nvar $nghost $nruns
#Outputs 
#nvars nx1 nx2 nx3 total_cells nruns \
#     time_packing time_per_kernel_packing cell_cycles_per_second_packing \
#     time_unpacking time_per_kernel_unpacking cell_cycles_per_second_unpacking

```

The script `scripts/bash/populate_timings.sh` generates timings for grid sizes `8^3` to `256^3`
