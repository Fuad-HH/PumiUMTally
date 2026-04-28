/**
 * @brief Implementation of the PumiTally class
 * @details It just wraps the PumiTallyImpl class to keep the interface
 * separate using the PIMPL approach.
 */

#include "PumiTally.h"
#include "PumiTallyImpl.h"

#include <chrono>
#include <memory>
#include <string>

namespace pumitally {

PumiTally::~PumiTally() {
  pimpl_.reset(nullptr);
  Kokkos::finalize();
}

PumiTally::PumiTally(const std::string &mesh_filename,
                     const int32_t num_particles, int &argc, char **&argv)
    : pimpl_(std::make_unique<PumiTallyImpl>(mesh_filename, num_particles, argc,
                                             argv)) {}

void PumiTally::CopyInitialPosition(double *init_particle_positions,
                                    const std::int32_t size) const {
  const auto start_time = std::chrono::steady_clock::now();

  pimpl_->CopyInitialPositionToBuffer(init_particle_positions, size);

  const std::chrono::duration<double> elapsed_seconds =
      std::chrono::steady_clock::now() - start_time;
  pimpl_->tally_times.initialization_time += elapsed_seconds.count();
}

void PumiTally::MoveToNextLocation(double *particle_origin,
                                   double *particle_destinations,
                                   int8_t *flying, double *weights,
                                   const std::int32_t size) const {
  const auto start_time = std::chrono::steady_clock::now();

  pimpl_->MoveToNextLocation(particle_origin, particle_destinations, flying,
                             weights, size);

  const std::chrono::duration<double> elapsed_seconds =
      std::chrono::steady_clock::now() - start_time;
  pimpl_->tally_times.total_time_to_tally += elapsed_seconds.count();
}

void PumiTally::WriteTallyResults() const {
  const auto start_time = std::chrono::steady_clock::now();

  pimpl_->WriteTallyResults();

  const std::chrono::duration<double> elapsed_seconds =
      std::chrono::steady_clock::now() - start_time;
  pimpl_->tally_times.vtk_file_write_time += elapsed_seconds.count();
  pimpl_->tally_times.PrintTimes();
}

} // namespace pumitally
