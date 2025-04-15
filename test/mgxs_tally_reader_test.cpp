//
// Created by Fuad Hasan on 3/21/25.
//

#include "MultiGroupXS.h"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <catch2/catch_test_macros.hpp>


TEST_CASE("Test reader") {
    Kokkos::initialize();
    {
        std::string filename = "/lore/hasanm4/wsources/openmc_pumi/pumitallyopenmc/test/assets/mgxs.h5";
        auto mgxs = read_mgxs(filename);


        printf("=>-------------- Test Reader --------------<=\n");
        printf("Number of groups: %d\n", mgxs.nEnergyGroups);
        REQUIRE(mgxs.nEnergyGroups == 2);

        REQUIRE(mgxs.materialNames.size() == 1);
        REQUIRE(mgxs.materialXSs.size() == 1);
        REQUIRE(mgxs.materialNames[0] == std::string("steel"));

        REQUIRE(mgxs.energyGroupEdges.size() == 3);
        auto energyGroupEdges_host = Omega_h::HostWrite<Omega_h::Real>(mgxs.energyGroupEdges);
        std::vector<double> expected_edges = {0, 0.75, 1.2};
        double margin = 1e-2;
        for (int i = 0; i < mgxs.nEnergyGroups + 1; ++i) {
            printf("Energy group edge %d: %f\n", i, energyGroupEdges_host[i]);
            REQUIRE_THAT(energyGroupEdges_host[i], Catch::Matchers::WithinAbs(expected_edges[i], margin));
        }

        REQUIRE(mgxs.materialXSs[0].isFissionable == false);
        REQUIRE(mgxs.materialXSs[0].order == 0);
        REQUIRE(mgxs.materialXSs[0].temperatures.size() == 1);
        REQUIRE(mgxs.materialXSs[0].temperatures[0] == std::string("294K"));


        REQUIRE(mgxs.materialXSs[0].crossSections.size() == 1);
        auto absorption_host = Omega_h::HostWrite<Omega_h::Real>(mgxs.materialXSs[0].crossSections[0].absorption);
        auto total_host = Omega_h::HostWrite<Omega_h::Real>(mgxs.materialXSs[0].crossSections[0].total);
        auto scatter_host = Omega_h::HostWrite<Omega_h::Real>(mgxs.materialXSs[0].crossSections[0].scatter_matrix);

        REQUIRE(absorption_host.size() == 2);
        REQUIRE(total_host.size() == 2);
        REQUIRE(scatter_host.size() == 4);

        std::vector<double> expected_absorption = {0, 1};
        std::vector<double> expected_total = {100, 33.3333};
        std::vector<double> expected_scatter = {100, 100, 100, 100};

        for(int i=0; i<absorption_host.size(); ++i) {
            printf("Absorption %d: %f\n", i, absorption_host[i]);
            printf("Total %d: %f\n", i, total_host[i]);
            REQUIRE_THAT(absorption_host[i], Catch::Matchers::WithinAbs(expected_absorption[i], margin));
            REQUIRE_THAT(total_host[i], Catch::Matchers::WithinAbs(expected_total[i], margin));
        }
        for (int i=0; i<scatter_host.size(); ++i) {
            printf("Scatter %d: %f\n", i, scatter_host[i]);
            REQUIRE_THAT(scatter_host[i], Catch::Matchers::WithinAbs(expected_scatter[i], margin));
        }
    }
    Kokkos::finalize();
}
