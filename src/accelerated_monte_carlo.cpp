//
// Created by Fuad Hasan on 4/30/25.
//

#include "physics/PumiTransport.h"
#include "pumitallyopenmc/pumipic_particle_data_structure.h"
#include <iostream>

#define max_steps 3

int main(int argc, char *argv[]) {
  Kokkos::initialize();
  {
    random_pool_t rpool(1);
    if (argc < 4) {
      std::cerr << "Usage: " << argv[0] << " <mesh> <n particles> <mgxs file>"
                << std::endl;
      return 1;
    }
    std::string mesh_file = argv[1];
    int n_particles = std::stoi(argv[2]);
    std::string mgxs_file = argv[3];
    printf("Simulating %d particles in mesh %s\n", n_particles,
           mesh_file.c_str());
    printf("\n\n=>=========================== Initial Setup "
           "=========================================\n\n");

    PumiTransport transportObject(
        mesh_file, mgxs_file, n_particles, argc, argv,
        {std::make_unique<Box>(0.1, 0.5, 0.1, 0.5, 0.1, 0.5), 1.0});
    transportObject.initializeSource();
    pumiinopenmc::PumiTally tally(mesh_file, n_particles, argc, argv);

    printf("\n\n=>=========================== Simulation "
           "=========================================\n\n");
    tally.initialize_particle_location(transportObject.particlePosition);

    for (int i = 0; i < max_steps; ++i) {
      printf("Iteration %d\n", i);
      if (!transportObject.areParticlesAlive()) {
        printf("=>=========================== All particles are dead "
               "=========================================\n");
        break;
      }

      transportObject.nextCollision(rpool);
      tally.move_to_next_location(transportObject.particlePosition,
                                  transportObject.particleWeight);
    }

    printf("\n\n=>=========================== Finalizing Simulation "
           "=========================================\n\n");
    tally.write_pumi_tally_mesh();
    printf("=>=========================== Simulation Complete "
           "=========================================\n");
  }
  Kokkos::finalize();
}
