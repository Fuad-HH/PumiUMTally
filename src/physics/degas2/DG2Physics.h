//
// Created by Fuad Hasan on 6/20/25.
//

#ifndef PUMITALLYOPENMC_DG2PHYSICS_H
#define PUMITALLYOPENMC_DG2PHYSICS_H

#define SEED 12345

#include "DG2CrossSection.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

typedef Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>
    random_pool_t;

struct ParticleInfo {
  double position[3];  // Position in space (x, y, z)
  double direction[3]; // Direction vector (unit vector)
  double weight;
  int energy_group; // Energy group *index*
};

struct FieldInfo {
  double electron_temperature;
  double ion_temperature;
  double electron_density;
  double ion_density;
  double bulk_flow_velocity[3];
};

class DG2Physics {
public:
  DG2Physics(const std::string &cross_section_file, const int num_particles,
             const int seed = SEED)
      : random_pool(seed), cross_section(cross_section_file) {
    // Initialize particle velocities
    particle_velocities =
        Kokkos::View<double *[3]>("particle_velocities", num_particles);
  }

  DG2Physics(const DG2CrossSection cross_section, const int num_particles,
             const int seed = SEED)
      : random_pool(seed), cross_section(cross_section) {
    // Initialize particle velocities
    particle_velocities =
        Kokkos::View<double *[3]>("particle_velocities", num_particles);
  }

  // sample a new distance and update the particle's position but does not
  // change direction
  KOKKOS_FUNCTION
  void sample_collision_distance(ParticleInfo &particle_info,
                                 const FieldInfo &field_info) const {
    // example of sampling a random number
    auto rand_gen = random_pool.get_state();
    double x = rand_gen.drand(0., 1.);
    random_pool.free_state(rand_gen);

    double y =
        x * cross_section.sigma_a(0, 0, 0); // Example usage of cross_section
  }

  // collision event
  KOKKOS_FUNCTION
  void collide_particle(ParticleInfo &particle_info,
                        const FieldInfo &field_info) const {}
  random_pool_t random_pool;
  DG2CrossSection cross_section;

  Kokkos::View<double *[3]> particle_velocities;
};
#endif // PUMITALLYOPENMC_DG2PHYSICS_H
