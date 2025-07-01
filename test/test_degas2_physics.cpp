//
// Created by Fuad Hasan on 6/20/25.
//

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "DG2Physics.h"
#include <Kokkos_Core.hpp>
#include <fstream>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    int numParticles = 10;

    Kokkos::View<Omega_h::Real ***> sigma_t_;            // mat, T, g
    Kokkos::View<Omega_h::Real ***> sigma_a_;            // mat, T, g
    Kokkos::View<Omega_h::Real ****> scattering_matrix_; // mat, T, g, g
    Kokkos::View<Omega_h::Real ***> sigma_s_;            // mat, T, g
    DG2CrossSection crossSection(sigma_t_, sigma_a_, scattering_matrix_,
                                 sigma_s_);
    DG2Physics physics(crossSection, numParticles);

    Kokkos::View<ParticleInfo *> particles("particles", numParticles);
    Kokkos::View<FieldInfo *> fields("fields", numParticles);
    Kokkos::parallel_for(
        "initialize_particles", numParticles, KOKKOS_LAMBDA(int i) {
          particles(i).position[0] = 0;
          particles(i).position[1] = 0;
          particles(i).position[2] = 0;
          particles(i).direction[0] = 1;
          particles(i).direction[1] = 0;
          particles(i).direction[2] = 0;
          particles(i).energy_group = 1;
          particles(i).weight = 1.0;

          fields(i).electron_density = 1.0;
          fields(i).ion_density = 1.0;
          fields(i).electron_temperature = 1.0;
          fields(i).ion_temperature = 1.0;
          // and others like this
        });

    // Call the function to be tested
	Kokkos::parallel_for(
        "run physics", numParticles, KOKKOS_LAMBDA(int i) {
          physics.collide_particle(particles(i), fields(i));
        });
    Kokkos::parallel_for(
        "run physics", numParticles, KOKKOS_LAMBDA(int i) {
          physics.sample_collision_distance(particles(i), fields(i));
        });
  }
  while (int n = 0; n<numParticles, n++) {
  	std::ofstream outfile("Log.txt");

	outfile << "Particle #" << n << std::endl;
	outfile << "Energy: " << particles(n).energy_group << ", Weight: " << particles(n).weight << std::endl;
	outfile << "Position: " << particles(n).position[0] << ", " << particles(n).position[1] << ", " << particles(n).position[2] << std::endl;
	outfile << "Direction: " << particles(n).direction[0] << ", " << particles(n).direction[1] << ", " << particles(n).direction[2] << std::endl;
    outfile << endl;
  }
  Kokkos::finalize();
  return 0;
}
