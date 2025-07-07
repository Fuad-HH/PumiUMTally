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
    int numParticles = 1000;

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
          particles(i).direction[0] = 1.0;
          particles(i).direction[1] = 0;
          particles(i).direction[2] = 0;
          particles(i).energy_group = 3.0;
          particles(i).weight = 1.0;
		  particles(i).particle_index = i;

          fields(i).electron_density = 1.0e13;
          fields(i).ion_density = 1.0e13;
          fields(i).electron_temperature = 1000.0;
          fields(i).ion_temperature = 1000.0;
          // and others like this
        });

    // Call the function to be tested
	Kokkos::parallel_for(
        "run physics", numParticles, KOKKOS_LAMBDA(int i) {
          physics.sample_collision_distance(particles(i), fields(i));
		  physics.collide_particle(particles(i), fields(i));
        });
	auto output = create_mirror_view(particles);
	Kokkos::deep_copy(output, particles);
	std::ofstream outfile("Log.txt");

	double l {0};
	for (int i=0; i < numParticles; ++i) {
		l += output(i).position[0];
	}
	l /= numParticles;
	outfile << "Average Distance (cm): " << l << std::endl;

	double varl {0};
	for (int i=0; i < numParticles; ++i) {
		varl += (output(i).position[0] - l)*(output(i).position[0] - l);
	}
	varl /= (numParticles - 1);
	double sdl {sqrt(varl)};
	outfile << "Standard Deviation of Distance (cm): " << sdl << std::endl;

	double ux {0};
	for (int i=0; i < numParticles; ++i) {
		ux += output(i).direction[0];
	}
	ux /= numParticles;
	outfile << "Mean x Direction: " << ux << std::endl;

	double varux;
	for (int i=0; i < numParticles; ++i) {
		varux += (output(i).direction[0] - ux)*(output(i).direction[0] - ux);
	}
	varux /= (numParticles - 1);
	double sdux {sqrt(varux)};
	outfile << "Standard Deviation of Mean x Direction: " << sdux << std::endl;


	outfile << "Particle #,Energy(eV),Weight,X(cm),Y(cm),Z(cm),X_Dir,Y_Dir,Z_Dir" << std::endl;

	for (int i = 0; i < numParticles; ++i) {
		outfile << i << ",";
		outfile << output(i).energy_group << ",";
		outfile << output(i).weight << ",";
		outfile << output(i).position[0] << ",";
		outfile << output(i).position[1] << ",";
		outfile << output(i).position[2] << ",";
		outfile << output(i).direction[0] << ",";
		outfile << output(i).direction[1] << ",";
		outfile << output(i).direction[2] << std::endl;
	}


  }
  Kokkos::finalize();
  return 0;
}
