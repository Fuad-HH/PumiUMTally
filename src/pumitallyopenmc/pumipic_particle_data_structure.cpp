#include "pumipic_particle_data_structure.h"
#include "pumitally_impl.hpp"

namespace pumiinopenmc {
PumiTally::PumiTally(std::string &mesh_filename, int64_t num_particles,
                     int &argc, char **&argv)
    : pimpl(std::make_unique<PumiTallyImpl>(mesh_filename, num_particles, argc,
                                            argv)) {}

void PumiTally::initialize_particle_location(double *init_particle_positions,
                                             int64_t size) {
  auto start_time = std::chrono::steady_clock::now();

  pimpl->initialize_particle_location(init_particle_positions, size);

  std::chrono::duration<double> elapsed_seconds =
      std::chrono::steady_clock::now() - start_time;
  pimpl->tally_times.initialization_time += elapsed_seconds.count();
}

void PumiTally::move_to_next_location(double *particle_destinations,
                                      int8_t *flying, double *weights,
                                      int *groups, int *material_ids,
                                      int64_t size) {
  auto start_time = std::chrono::steady_clock::now();

  pimpl->move_to_next_location(particle_destinations, flying, weights, groups,
                               material_ids, size);

  std::chrono::duration<double> elapsed_seconds =
      std::chrono::steady_clock::now() - start_time;
  pimpl->tally_times.total_time_to_tally += elapsed_seconds.count();
}

void PumiTally::write_pumi_tally_mesh() {
  auto start_time = std::chrono::steady_clock::now();

  pimpl->write_pumi_tally_mesh();

  std::chrono::duration<double> elapsed_seconds =
      std::chrono::steady_clock::now() - start_time;
  pimpl->tally_times.vtk_file_write_time += elapsed_seconds.count();
  pimpl->tally_times.print_times();
}
} // namespace pumiinopenmc
