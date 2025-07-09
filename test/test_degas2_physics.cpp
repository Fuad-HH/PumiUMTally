//
// Created by Fuad Hasan on 6/20/25.
//

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "DG2Physics.h"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Test Degas2 Physics Functions"){
  Kokkos::initialize();

  {
    int numParticles = 1000;
	double energy = 3.0;

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
          particles(i).energy_group = energy;
          particles(i).weight = 1.0;
		  particles(i).particle_index = i;

          fields(i).electron_density = 1.0e13;
          fields(i).ion_density = 1.0e13;
          fields(i).electron_temperature = 1000.0;
          fields(i).ion_temperature = 1000.0;
          // and others like this
        });

	// Set the particle energy view
	Kokkos::parallel_for(
		"Set energies", numParticles, KOKKOS_LAMBDA(int i) {
		  physics.set_energy(particles(i), energy);
	});

    // Call the function to be tested
	Kokkos::parallel_for(
        "run physics", numParticles, KOKKOS_LAMBDA(int i) {
          physics.sample_collision_distance(particles(i), fields(i));
		  physics.collide_particle(particles(i), fields(i));
        });
	auto output = create_mirror_view(particles);
	Kokkos::deep_copy(output, particles);
	//std::ofstream outfile("Log.txt");

	//These 4 functions should return the commented value for SEED=12345

	double l {0};
	for (int i=0; i < numParticles; ++i) {
		l += output(i).position[0];
	}
	l /= numParticles;
	//outfile << "Average Distance (cm): " << l << std::endl; //2.41

	double varl {0};
	for (int i=0; i < numParticles; ++i) {
		varl += (output(i).position[0] - l)*(output(i).position[0] - l);
	}
	varl /= (numParticles - 1);
	double sdl {sqrt(varl)};
	//outfile << "Standard Deviation of Distance (cm): " << sdl << std::endl; //2.38

	double ux {0};
	for (int i=0; i < numParticles; ++i) {
		ux += output(i).direction[0];
	}
	ux /= numParticles;
	//outfile << "Mean x Direction: " << ux << std::endl; //0.0159

	double varux;
	for (int i=0; i < numParticles; ++i) {
		varux += (output(i).direction[0] - ux)*(output(i).direction[0] - ux);
	}
	varux /= (numParticles - 1);
	double sdux {sqrt(varux)};

	/*
	outfile << "Standard Deviation of Mean x Direction: " << sdux << std::endl; //0.579


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
	*/
  	REQUIRE_THAT(l, Catch::Matchers::WithinAbs(2.40,.03));
  	REQUIRE_THAT(sdl, Catch::Matchers::WithinAbs(2.40,.03));
  	REQUIRE_THAT(ux, Catch::Matchers::WithinAbs(0.0,.03));
  	REQUIRE_THAT(sdux, Catch::Matchers::WithinAbs(0.577,.03));
  }
  Kokkos::finalize();

}

TEST_CASE("Test Degas2 Physics Particle Track Until Destroyed"){
  Kokkos::initialize();

  {
    int numParticles = 1;
	double energy = 3.0;

    Kokkos::View<Omega_h::Real ***> sigma_t_;            // mat, T, g
    Kokkos::View<Omega_h::Real ***> sigma_a_;            // mat, T, g
    Kokkos::View<Omega_h::Real ****> scattering_matrix_; // mat, T, g, g
    Kokkos::View<Omega_h::Real ***> sigma_s_;            // mat, T, g
    DG2CrossSection crossSection(sigma_t_, sigma_a_, scattering_matrix_,
                                 sigma_s_);
    DG2Physics physics(crossSection, numParticles);

    Kokkos::View<ParticleInfo *> particles("particles", numParticles);
    Kokkos::View<FieldInfo *> fields("fields", numParticles);
	Kokkos::View<ParticleInfo *> track("track", 1);

    Kokkos::parallel_for(
        "initialize_particles", numParticles, KOKKOS_LAMBDA(int i) {
          particles(i).position[0] = 0;
          particles(i).position[1] = 0;
          particles(i).position[2] = 0;
          particles(i).direction[0] = 1.0;
          particles(i).direction[1] = 0;
          particles(i).direction[2] = 0;
          particles(i).energy_group = energy;
          particles(i).weight = 1.0;
		  particles(i).particle_index = i;

          fields(i).electron_density = 1.0e13;
          fields(i).ion_density = 1.0e13;
          fields(i).electron_temperature = 1000.0;
          fields(i).ion_temperature = 1000.0;
          // and others like this
        });

	// Set the particle energy view and randomize direction and energy under the Maxwelllian
	Kokkos::parallel_for(
		"Set energies", numParticles, KOKKOS_LAMBDA(int i) {
		  physics.set_energy(particles(i), energy);
		  physics.collide_particle(particles(i), fields(i));
	});

	//Reset weight
	Kokkos::parallel_for(
        "initialize_particles", numParticles, KOKKOS_LAMBDA(int i) {
          particles(i).weight = 1.0;
        });

    // Call the function to be tested
	Kokkos::parallel_for(
        "run physics", numParticles, KOKKOS_LAMBDA(int i) {
		  int j = 0;
		  while(particles(i).weight > 0.0001){
			if (j != 0){
				Kokkos::resize(track, (j+1));
			}
          	physics.sample_collision_distance(particles(i), fields(i));
		  	physics.collide_particle(particles(i), fields(i));
			track(j) = particles(i);
			j++;
		  }
        });
	auto output = create_mirror_view(track);
	Kokkos::deep_copy(output, track);

	int length = output.extent(0);

	std::ofstream outfile("Log.txt");
	outfile << "Particle #,Energy(eV),Weight,X(cm),Y(cm),Z(cm),X_Dir,Y_Dir,Z_Dir" << std::endl;

	for (int i = 0; i < length; ++i) {
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
  	REQUIRE_THAT(output(length-1).weight, Catch::Matchers::WithinAbs(0,.03));

  }
  Kokkos::finalize();

}