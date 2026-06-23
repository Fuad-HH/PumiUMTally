/**
 * @brief User Interface for PUMI-Tally
 */

#ifndef PUMITALLY_PUMITALLY_H
#define PUMITALLY_PUMITALLY_H

#include <cstdint>
#include <memory>
#include <vector>

/**
 * @brief PUMI-Tally Namespace
 * @details Provides API to PUMI-PiC for Parallel Unstructured Mesh
 * Tally Operations
 */
namespace pumitally {
/**
 * @brief Provides the functor to execute when particle reaches element boundary
 * @details
 * Evaluates flux and its auxiliary operations after crossing element
 * boundaries.
 */
struct ParticleAtElemBoundary;
struct PumiTallyImpl;

/**
 * @brief PUMI-Tally Interface Class
 * @details Provides user interface to call directly from physics application.
 * All functions take C++ builtin variable types for robustness. All PUMI-PiC
 * and Omega_h mesh related data are held inside the @ref PUMITallyImpl class.
 *
 * @see PUMITallyImpl
 */
class PumiTally {
public:
  /**
   * @brief Read mesh and initialize particles
   * @details Calls the @ref pimpl constructor to allocate particle DS and other
   * auxiliary arrays
   * @param mesh_filename Omega_h mesh filename (*.osh file)
   * @param num_particles Actual number of particles
   * @param argc Program argc
   * @param argv Program argv
   *
   * @note The mesh file name should end like `*.osh`, not `*.osh/`.
   * Autocomplete may sometimes add a `/` with the mesh name since Omega_h
   * meshes are stored as directories.
   * @n `argc` and `argv` are takes for MPI and Kokkos inputs.
   */
  PumiTally(const std::string &mesh_filename, int32_t num_particles, int &argc,
            char **&argv);

  /**
   * Perform the is_initial_track search
   * @param init_particle_positions Positions of the particles flattened as x1,
   * y1, z1, x2, y2, ...
   * @param size Number of particles
   *
   * @details
   * Monte Carlo physics codes generally samples the origin points based on the
   * users' choice and physics is involved in it. PUMI-Tally always needs to
   * know which mesh element they are currently in. Therefore, an initial search
   * is done to find the starting position of the particles. This can be only
   * called once after the source initialization.
   */
  void CopyInitialPosition(double *init_particle_positions,
                           std::int32_t size) const;

  /**
   * Track particles to a new location and tally
   *
   * @param particle_origin Current particle positions flattened as x1, y1, z1,
   * x2, y2, ...
   * @param particle_destinations Particle destinations flattened origin
   * locations
   * @param flying If the particles are flying in this step 1-flying 0-stopped
   * @param weights Weight of particle (multiplied when tallying), usually [0,1]
   * @param size Number of particles (use int64_t for large particle counts)
   *
   * @details
   * It first moves the particles to the origin locations. This step is
   * necessary because sometimes particles get reincarnated (for example in
   * OpenMC, particles get absorbed and resampled to a new location) and show up
   * at a new location. Then they move to the destination. They are not tallied
   * when moving to the origin (first step).
   */
  void MoveToNextLocation(double *particle_origin,
                          double *particle_destinations, int8_t *flying,
                          double *weights, int64_t size) const;

  /**
   * @brief Write the mesh tally to a VTK file
   * @details Normalized by element volumes and total number of particles
   */
  void WriteTallyResults() const;

  /**
   * @brief Register an element-based multi-dimensional tally (call once)
   * @param number_of_non_spatial_filter_bins Number of bins per filter
   *        (e.g., [2, 4] for 2 energy bins and 4 polar angle bins).
   *        The number of filters is inferred from the vector size.
   *
   * @details The resulting tally array is a flat Kokkos View of size
   *   nelem × bins[0] × bins[1] × ...
   * For example with [2, 4]: nelem × 8 entries.
   * This method can only be called once.
   */
  void AddElementTally(
      std::vector<unsigned int> number_of_non_spatial_filter_bins);

  /**
   * @brief Register a node-based (vertex) multi-dimensional tally (call once)
   * @param number_of_non_spatial_filter_bins Number of bins per filter.
   *        The number of filters is inferred from the vector size.
   *
   * @details The resulting tally array is a flat Kokkos View of size
   *   nvertices × bins[0] × bins[1] × ...
   * This method can only be called once.
   */
  void AddNodeTally(
      std::vector<unsigned int> number_of_non_spatial_filter_bins);

  /**
   * @brief Set per-particle filter bin assignments for the current step
   * @param bins Flat array of size nparticles * n_filters.
   *        For particle p, bins[p * n_filters + f] is the bin index for
   *        filter dimension f. Must be called before MoveToNextLocation.
   *
   * @details
   * Example with [2, 4] filter bins (n_filters = 2):
   *   bins size = nparticles * 2
   *   bins[pid * 2 + 0] = energy bin (0 or 1)
   *   bins[pid * 2 + 1] = angle bin (0, 1, 2, or 3)
   */
  void UpdateFilterBins(std::vector<unsigned int> bins);

  /**
   * @brief (TODO: TEMPORARY) Enable specular (equal-angle) reflection at domain boundaries
   * @details When a particle hits the outer boundary, its direction is
   * reflected using: reflected = incident - 2·dot(incident,normal)·normal.
   * The particle stays in the same element and continues with the reflected
   * direction. By default, vacuum boundary condition is used (particle leaves).
   */
  void SetReflectiveBoundaryCondition();

  /**
   * @brief PUMI-Tally Destructor
   * @details Call at the end of tally operations to avoid memory leaks.
   *
   * @see Pumitally
   */
  ~PumiTally();

private:
  std::unique_ptr<PumiTallyImpl> pimpl_; //!< @ref PumiTallyImpl holder
};
} // namespace pumitally

#endif // PUMITALLY_PUMITALLY_H
