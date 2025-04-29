//
// Created by Fuad Hasan on 3/21/25.
//

#include "MultiGroupXS.h"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <catch2/catch_test_macros.hpp>


TEST_CASE("Test reader") {
    Kokkos::initialize();
    {
        std::string filename = "assets/mgxs.h5";
        MultiGroupXS mgxs(filename);
        mgxs.print();


        printf("=>-------------- Test Reader --------------<=\n");
        printf("Number of groups: %d\n", mgxs.getNumEnergyGroups());
        REQUIRE(mgxs.getNumEnergyGroups() == 2);

        auto materialNames = mgxs.getMaterialNames();
        REQUIRE(materialNames.size() == 1);
        REQUIRE(materialNames[0] == std::string("steel"));

        auto energyGroupEdges = mgxs.getEnergyGroupEdges();
        REQUIRE(energyGroupEdges.size() == 3);
        auto energyGroupEdges_host = Omega_h::HostRead<Omega_h::Real>(energyGroupEdges);
        std::vector<double> expected_edges = {0, 0.75, 1.2};
        double margin = 1e-2;
        for (int i = 0; i < mgxs.getNumEnergyGroups() + 1; ++i) {
            printf("Energy group edge %d: %f\n", i, energyGroupEdges_host[i]);
            REQUIRE_THAT(energyGroupEdges_host[i], Catch::Matchers::WithinAbs(expected_edges[i], margin));
        }


        auto temperatureNames = mgxs.getTemperatureNames();
        REQUIRE(temperatureNames.size() == 1);
        REQUIRE(temperatureNames[0] == std::string("294K"));


        auto absorption_host = Kokkos::create_mirror_view(mgxs.getSigmaA());
        auto total_host = Kokkos::create_mirror_view(mgxs.getSigmaT());
        auto scatter_host = Kokkos::create_mirror_view(mgxs.getSigmaS());
        Kokkos::deep_copy(absorption_host, mgxs.getSigmaA());
        Kokkos::deep_copy(total_host, mgxs.getSigmaT());
        Kokkos::deep_copy(scatter_host, mgxs.getSigmaS());

        REQUIRE(absorption_host.extent(0) == 1); // number of materials
        REQUIRE(absorption_host.extent(1) == 1); // number of temperatures
        REQUIRE(absorption_host.extent(2) == mgxs.getNumEnergyGroups()); // number of energy groups
        REQUIRE(total_host.extent(0) == 1); // number of materials
        REQUIRE(total_host.extent(1) == 1); // number of temperatures
        REQUIRE(total_host.extent(2) == mgxs.getNumEnergyGroups()); // number of energy groups

        REQUIRE(scatter_host.extent(0) == 1); // number of materials
        REQUIRE(scatter_host.extent(1) == 1); // number of temperatures
        REQUIRE(scatter_host.extent(2) == mgxs.getNumEnergyGroups()); // number of energy groups
        REQUIRE(scatter_host.extent(3) == mgxs.getNumEnergyGroups()); // number of energy groups

        std::vector<double> expected_absorption = {0, 1};
        std::vector<double> expected_total = {100, 33.3333};
        std::vector<double> expected_scatter = {100, 100, 100, 100};

        for(int i=0; i<2; ++i) {
            printf("Absorption %d: %f\n", i, absorption_host(0,0, i));
            printf("Total %d: %f\n", i, total_host(0, 0, i));
            REQUIRE_THAT(absorption_host(0,0,i), Catch::Matchers::WithinAbs(expected_absorption[i], margin));
            REQUIRE_THAT(total_host(0,0,i), Catch::Matchers::WithinAbs(expected_total[i], margin));
        }
        for (int i=0; i<2; ++i) {
            for (int j=0; j<2; ++j) {
                printf("Scatter %d %d: %f\n", i, j, scatter_host(0, 0, i, j));
                REQUIRE_THAT(scatter_host(0, 0, i, j), Catch::Matchers::WithinAbs(expected_scatter[i*2+j], margin));
            }
        }
    }
    Kokkos::finalize();
}
