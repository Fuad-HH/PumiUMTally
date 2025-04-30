//
// Created by Fuad Hasan on 4/29/25.
//

#include "PumiTransport.h"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test Transport") {
  Kokkos::initialize();
  {
    random_pool_t rpool(1);
    int argc = 1;
    char *argv[1];
    auto defaultBoxSourceTransport =
        PumiTransport("notused", "assets/mgxs.h5", 10000, argc, argv);
    defaultBoxSourceTransport.initializeSource();
    // TODO: remove this line
    defaultBoxSourceTransport.writePositionsForGNUPlot("../build/test_box.dat");

    auto sphereSourceTransport =
        PumiTransport("notused", "assets/mgxs.h5", 10, argc, argv,
                      {std::make_unique<Sphere>(3.0, 1, 2, 3), 1.0}, rpool);
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
      printf("Particle %d: energy_host %f  group %d \n", i, energy_host(i),
             energy_group_host(i));
      REQUIRE(energy_group_host(i) == 1);
    }

    // do a test for the particle positions
    sphereSourceTransport.nextCollision(rpool);

    // particle energies and groups after collision
    auto particle_energies_host =
        Kokkos::create_mirror_view(sphereSourceTransport.particleEnergy);
    auto particle_mat_temp_group_host =
        Kokkos::create_mirror_view(sphereSourceTransport.matTempEg);
    Kokkos::deep_copy(particle_energies_host,
                      sphereSourceTransport.particleEnergy);
    Kokkos::deep_copy(particle_mat_temp_group_host,
                      sphereSourceTransport.matTempEg);

    for (int i = 0; i < sphereSourceTransport.getNumParticles(); ++i) {
      printf("Partcle %d: energy %f  group %d \n", i, particle_energies_host(i),
             particle_mat_temp_group_host(i, 2));
    }

    // new positions
    auto particle_positions_host =
        Kokkos::create_mirror_view(sphereSourceTransport.particlePosition);
    auto particle_directions_host =
        Kokkos::create_mirror_view(sphereSourceTransport.particleDirection);
    Kokkos::deep_copy(particle_positions_host,
                      sphereSourceTransport.particlePosition);
    Kokkos::deep_copy(particle_directions_host,
                      sphereSourceTransport.particleDirection);

    for (int i = 0; i < sphereSourceTransport.getNumParticles(); ++i) {
      printf("Particle %d: position %f %f %f  direction %f %f %f \n", i,
             particle_positions_host(3 * i), particle_positions_host(3 * i + 1),
             particle_positions_host(3 * i + 2),
             particle_directions_host(3 * i),
             particle_directions_host(3 * i + 1),
             particle_directions_host(3 * i + 2));
    }
  }
  Kokkos::finalize();
}
