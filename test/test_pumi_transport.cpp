//
// Created by Fuad Hasan on 4/29/25.
//

#include "PumiTransport.h"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test Transport") {
  Kokkos::initialize();
  {
    int argc = 1;
    char *argv[1];
    auto defaultBoxSourceTransport =
        PumiTransport("notused", "assets/mgxs.h5", 10000, argc, argv);
    defaultBoxSourceTransport.initializeSource();
    // TODO: remove this line
    defaultBoxSourceTransport.writePositionsForGNUPlot("../build/test_box.dat");

    auto sphereSourceTransport =
        PumiTransport("notused", "assets/mgxs.h5", 10000, argc, argv,
                      {std::make_unique<Sphere>(3.0, 1, 2, 3), 1.0});
    sphereSourceTransport.initializeSource();
    // TODO: remove this line
    sphereSourceTransport.writePositionsForGNUPlot("../build/test_sphere.dat");

    // Initial Energy Group
    auto energy_group_host =
        Kokkos::create_mirror_view(sphereSourceTransport.particleEnergyGroup);
    auto energy_host =
        Kokkos::create_mirror_view(sphereSourceTransport.particleEnergy);
    Kokkos::deep_copy(energy_group_host,
                      sphereSourceTransport.particleEnergyGroup);
    Kokkos::deep_copy(energy_host, sphereSourceTransport.particleEnergy);
    REQUIRE(energy_group_host.size() ==
            sphereSourceTransport.getNumParticles());
    REQUIRE(energy_host.size() == sphereSourceTransport.getNumParticles());

    for (int i = 0; i < sphereSourceTransport.getNumParticles(); ++i) {
      printf("Particle %d: energy_host %f  group %d n", i, energy_host(i),
             energy_group_host(i));
      REQUIRE(energy_group_host(i) == 1);
    }
  }
  Kokkos::finalize();
}