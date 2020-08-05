#!/bin/bash

n_var=5
n_ghost=2

export OMP_PROC_BIND=spread 
export OMP_PLACES=threads


for name in "kokkos_buffer_packing";
do
    echo "" > $name-timings.dat
    for n_side_log2 in $(seq 3 9);
    do
        n_side=$(( 2**$n_side_log2 ))
        n_run=$(( 2**(15-$n_side_log2) ))
        echo "Running $name $n_side $n_side $n_side $n_var $n_ghost $n_run"
        tests/$name/$name $n_side $n_side $n_side $n_var $n_ghost $n_run >> $name-timings.dat
    done
done

