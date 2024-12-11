//
// Created by Fuad Hasan on 12/3/24.
//
#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>

TEST_CASE("Omega_h Host Array Test") {
    int argc = 0;
    char** argv = nullptr;
    auto omega_h_lib = Omega_h::Library(&argc, &argv);

    int size = 10;
    Omega_h::HostWrite<Omega_h::Real> test_host_array(size, 7.0, 0, "test_array");
    for (int i = 0; i < test_host_array.size(); i++) {
        printf("test_array[%d] = %f\n", i, test_host_array[i]);
        REQUIRE(test_host_array[i] == 7.0);
    }
    printf("\nPrinted the array...\n");

    Omega_h::Write<Omega_h::Real> test_device_array(test_host_array);
    Omega_h::Real properly_initialized = 0;

    Kokkos::Sum<Omega_h::Real> sum_reducer(properly_initialized);
    auto find_sum = KOKKOS_LAMBDA(const int i, Omega_h::Real& properly_initialized) {
        sum_reducer.join(properly_initialized, test_device_array[i]);
    };
    Kokkos::parallel_reduce(size, find_sum, sum_reducer);

    REQUIRE(properly_initialized == 7.0 * size);
}
