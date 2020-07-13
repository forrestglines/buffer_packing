add_executable(
    kokkos_buffer_packing
        kokkos_buffer_packing.cpp
)
target_link_libraries(kokkos_buffer_packing PUBLIC Kokkos::kokkos)
