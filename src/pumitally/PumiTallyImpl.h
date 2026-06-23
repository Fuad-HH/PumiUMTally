//
// Created by Fuad Hasan on 4/28/26.
//

#ifndef PUMITALLY_PUMITALLYIMPL_H
#define PUMITALLY_PUMITALLYIMPL_H

#include <ParticleTracer.tpp>
#include <pumipic_adjacency.tpp>
#include <pumipic_library.hpp>
#include <pumipic_mesh.hpp>

#include <vector>

namespace pumitally {

Omega_h::Reals GetCentroids(Omega_h::Mesh &mesh, bool add_tag = true);

/**
 *  Data structure to hold the timing information for different sections
 */
struct TallyTimes {
  double initialization_time = 0.0; //!< Time to read mesh and create DS
  double total_time_to_tally = 0.0; //!< Total time for data transfer and search
  double vtk_file_write_time = 0.0; //!< Time to write resulting VTK file

  /**
   * @brief Print the timing information in a readable format
   */
  void PrintTimes() const;
};

enum class SourceDistribution {
  UNIFORM, // Source uniformly distributed across the mesh
  EQUAL,   // Source at centroids of each element
  ZERO     // in the zeroth element centroid
};

/**
 * @brief PUMI-PiC Data structure Template
 * @details
 * Data:
 * @n   0-origin,
 * @n   1-destination,
 * @n   2-ID,
 * @n   3-in_advance_particle_queue,
 * @n   4-weight
 * @n   5-group
 */
using PPParticle =
    pumipic::MemberTypes<pumipic::Vector3d, pumipic::Vector3d, Omega_h::LO,
                         Omega_h::I16, Omega_h::Real, Omega_h::I16>;
using PPPS = pumipic::ParticleStructure<PPParticle>; //!< PUMI-PiC Particle DS
using PPExeSpace =
    Kokkos::DefaultExecutionSpace; //!< PUMI-PiC Default Execution Space

/**
 * @brief Specification for a set of non-spatial tally filters
 * @details Stores the filter bin counts. The number of filters is derived
 * from the size of bins_per_filter.
 */
struct TallySpec {
  std::vector<uint> bins_per_filter;      //!< Number of bins per filter
  uint total_filter_bins = 1;             //!< Product of bins_per_filter
  bool is_initialized = false;            //!< Whether this spec is valid

  TallySpec() = default;

  TallySpec(const std::vector<uint> &bins)
      : bins_per_filter(bins), is_initialized(true) {
    total_filter_bins = 1;
    for (auto b : bins_per_filter) {
      total_filter_bins *= b;
    }
  }

  uint GetNumFilters() const { return static_cast<uint>(bins_per_filter.size()); }
};

struct ParticleAtElemBoundary {
  /**
   * Allocates tally and other arrays
   * @param num_elements Number of mesh elements
   * @param num_vertices Number of mesh vertices (for node tallies)
   * @param capacity PUMI-PiC Particle DS capacity
   */
  ParticleAtElemBoundary(Omega_h::LO num_elements, Omega_h::LO num_vertices,
                         Omega_h::LO capacity);

  /**
   * @brief This operator is called by the ParticleTracer to do user defined
   * operations at element boundaries.
   * @details
   * This operator calls all the other functions defined in this struct:
   * - updatePrevXPoint
   * - evaluateFlux
   * - apply_boundary_condition
   * - move_to_next_element
   * @param mesh Omega_h mesh
   * @param ptcls PUMI-PiC Particle DS
   * @param elem_ids Current element ids of particles when tracking
   * @param next_elems Next element along particle trajectory
   * @param inter_faces ID of last intersected face
   * @param last_exit TODO find the difference with inter_faces
   * @param inter_points Particle intersection location of last face
   * @param ptcl_done If particle tracking is done for this step
   * @param origin_segment Origin locations segment
   * @param dest_segment Destination locations segment
   */
  void
  operator()(const Omega_h::Mesh &mesh, pumitally::PPPS *ptcls,
             const Omega_h::Write<Omega_h::LO> &elem_ids,
             const Omega_h::Write<Omega_h::LO> &next_elems,
             const Omega_h::Write<Omega_h::LO> &inter_faces,
             const Omega_h::Write<Omega_h::LO> &last_exit,
             const Omega_h::Write<Omega_h::Real> &inter_points,
             const Omega_h::Write<Omega_h::LO> &ptcl_done,
             decltype(ptcls->get<0>())
                 origin_segment, // NOLINT(performance-unnecessary-value-param)
             decltype(ptcls->get<1>()) dest_segment)
      const; // NOLINT(performance-unnecessary-value-param)
  /**
   * Save the current intersection points
   * @param xpoints Intersection points (flat: x0,y0,z0,x1,y1,z1,...)
   */
  void
  UpdatePreviousXPoints(const Omega_h::Write<Omega_h::Real> &xpoints) const;

  /**
   * Save particle origin points as previous intersection points
   * @param ptcls PUMI-PiC Particle DS
   * @details This is generally used to initialize the prev_xpoint array
   * with the starting positions.
   */
  void UpdatePreviousXPoints(PPPS *ptcls) const;

  /**
   * Calculate track-length estimated tally contributions
   * @param ptcls PUMI-PiC Particle DS
   * @param xpoints Current intersection points (flat: x0,y0,z0,x1,y1,z1,...)
   * @param elem_ids Current element ID
   * @param ptcl_done If particle tracking is done for this step
   *
   * @details
   * Calculates the track segment length inside the current element and
   * multiplies it with the particle weight before accumulating into the
   * multi-dimensional tally arrays.
   *
   * @see operator()
   */
  void EvaluateFlux(PPPS *ptcls, const Omega_h::Write<Omega_h::Real> &xpoints,
                    const Omega_h::Write<Omega_h::LO> &elem_ids,
                    const Omega_h::Write<Omega_h::LO> &ptcl_done) const;

  /**
   * Write tally results to a VTK file
   * @param full_mesh Omega_h mesh to write the tally results on
   * @param filename VTK file name
   *
   * @details Computes element volumes, attaches them as a tag, and writes
   * multi-dimensional element and node tally arrays to the mesh.
   */
  void FinalizeTallies(Omega_h::Mesh &full_mesh,
                       const std::string &filename) const;

  /**
   * Mark the tracking step as is_initial_track step
   * @details It turns off tallying for this step
   * @param is_initial If it is initial
   */
  void MarkAsInitial(bool is_initial);

  /**
   * @brief Register an element-based multi-dimensional tally (call once)
   * @param number_of_non_spatial_filter_bins Number of bins per filter
   *        (e.g., [2, 4] for 2 energy bins and 4 polar angle bins).
   *        The number of filters is inferred from the vector size.
   */
  void AddElementTally(
      const std::vector<uint> &number_of_non_spatial_filter_bins);

  /**
   * @brief Register a node-based (vertex) multi-dimensional tally (call once)
   * @param number_of_non_spatial_filter_bins Number of bins per filter.
   *        The number of filters is inferred from the vector size.
   */
  void AddNodeTally(const std::vector<uint> &number_of_non_spatial_filter_bins);

  // --- New: Filter bin management ---

  /**
   * @brief Set per-particle filter bin assignments for the current step
   * @param bins Flat host array: [pid * n_filters + dim] = bin_index
   *        Size must be capacity * n_filters where n_filters =
   *        bins_per_filter.size() from the registered tally.
   */
  void SetFilterBins(const std::vector<uint> &bins);

  // --- New: Boundary condition ---

  /**
   * @brief Boundary condition type
   */
  enum class BoundaryCondition {
    VACUUM,     // Particle leaves the domain, no reflection
    REFLECTIVE  // Specular reflection at domain boundaries
  };

  /**
   * @brief Set the boundary condition for particle-boundary interactions
   * @param bc The desired boundary condition
   * @param mesh Mesh to compute normals on if reflective (may be modified)
   */
  void SetBoundaryCondition(BoundaryCondition bc, Omega_h::Mesh &mesh);

  // --- New: Accessors for tally data ---

  /**
   * @brief Check whether any multi-dimensional tallies have been registered
   */
  bool HasMultiDimTally() const {
    return element_tally_spec.is_initialized ||
           node_tally_spec.is_initialized;
  }

  /**
   * @brief Get the registered filter specification
   */
  const TallySpec &GetTallySpec() const {
    if (element_tally_spec.is_initialized) {
      return element_tally_spec;
    }
    return node_tally_spec;
  }

  bool is_initial_track; //!< in is_initial_track run, tally is not accumulated
  Omega_h::LO nelem;                     //!< Number of mesh elements

  // --- Multi-dimensional tally arrays (flat Kokkos Views) ---
  // Logical layout: [spatial, bins[0], bins[1], ...] stored row-major flat.
  // Linear index: spatial_id * total_bins + flat_filter_idx
  TallySpec element_tally_spec;               //!< Spec for element tally
  TallySpec node_tally_spec;                  //!< Spec for node tally
  Kokkos::View<double *, PPExeSpace> element_tallies; //!< Flat: nelem * total_bins
  Kokkos::View<double *, PPExeSpace> node_tallies;    //!< Flat: nvert * total_bins
  bool multi_dim_tallies_active;              //!< Whether multi-D tallies are active

  // Once-only guards
  bool element_tally_called = false;
  bool node_tally_called = false;

  Omega_h::Write<Omega_h::Real> prev_xpoint; //!< Previous intersection point

  // --- Per-particle filter bin device array ---
  // Size: [capacity * n_filters], layout: [pid * n_filters + dim] = bin_index
  Omega_h::Write<Omega_h::LO> filter_bins_dev;
  uint active_n_filters;                      //!< Current number of filters in use

  // --- Device-accessible filter metadata (for use in device lambdas) ---
  // bins_per_filter[f] stored on device so EvaluateFlux can compute flat indices
  Omega_h::Write<Omega_h::LO> bins_per_filter_dev;
  // Precomputed strides: stride[f] = prod(bins[f+1..n-1]), last dimension = 1
  Omega_h::Write<Omega_h::LO> filter_strides_dev;
  // Total number of filter bin combinations (product of all bins_per_filter)
  Omega_h::Write<Omega_h::LO> total_filter_bins_dev;

  // --- Node tally support ---
  Omega_h::LO num_vertices;                   //!< Number of mesh vertices
  Omega_h::Read<Omega_h::LO> elem2vert;       //!< Element-to-vertex connectivity

  // --- Boundary condition ---
  BoundaryCondition boundary_condition;       //!< Active boundary condition
  Omega_h::Reals boundary_normals;            //!< Pre-computed boundary normals
                                              //!< (stored separately from mesh so
                                              //!<  they survive mesh copies)

  // temporary gabe merging variables
  // these will be removed after the operator functinality is merged to both
  Omega_h::Write<Omega_h::LO> last_exit_;
  Omega_h::Write<Omega_h::Real> alpha_;
};

/**
 * @brief PumiTallyImpl class
 * @details
 * This class is the implementation of the PumiTally interface.
 * It contains the data structures and methods to perform the tally operations.
 * @see PumiTally
 */
struct PumiTallyImpl {
  Omega_h::LO num_particles = 1e5; //!< Number of Particles
  std::string oh_mesh_filename;    //!< Omega_h mesh file name

  Omega_h::Library oh_lib; //!< Omega_h Library (Holds MPI Comm)
  Omega_h::Mesh full_mesh; //!< Full mesh before partition

  std::unique_ptr<pumipic::Library> pumipic_lib =
      nullptr; //!< PUMI-PiC Library (Holds Omega_h library)
  std::unique_ptr<pumipic::Mesh> p_picparts = nullptr; //!< Partitioned meshes
  std::unique_ptr<PPPS> pumipic_ptcls =
      nullptr; //!< PUMI-PiC Particle DS Instance

  long double pumipic_tol = 1e-8;      //!< Geometric comparison tolerance
  bool is_pumipic_initialized = false; //!< State of array allocations
  Omega_h::LO iter_count = 0; //!< Number of iterations for each move call
  double total_initial_weight =
      0.0; //!< Total is_initial_track weight (needed for normalization)

  std::unique_ptr<ParticleAtElemBoundary>
      p_pumi_particle_at_elem_boundary_handler; //!< Functor to call when
  //!< particles reach element
  //!< boundary
  std::unique_ptr<ParticleTracer<PPParticle, pumitally::ParticleAtElemBoundary>>
      p_particle_tracer; //!< PUMI-Pic Search Class Instance

  Omega_h::Write<Omega_h::Real>
      position_dev_buffer; //!< Particle coordinate buffer
  Omega_h::Write<Omega_h::I8>
      flying_dev_buffer; //!< Particle moving status buffer
  Omega_h::Write<Omega_h::Real> weights_dev_buffer; //!< Particle weight buffer

  TallyTimes tally_times; //!< Struct to hold times for different operations

  PumiTallyImpl(const std::string &mesh_filename, Omega_h::LO num_ptcls,
                int argc, char **argv,
                SourceDistribution source_dist = SourceDistribution::ZERO);

  ~PumiTallyImpl() = default;

  void InitializePUMIParticleStructure(Omega_h::Mesh &mesh);

  void LoadMeshAndInitParticles(int &argc, char **&argv);

  Omega_h::Mesh PartitionMesh();

  void InitPUMILibrary(int &argc, char **&argv);

  void SearchAndRebuild(bool initial, bool migrate = true) const;

  void ReadFullMesh(int &argc, char **&argv);

  void CopyInitialPositionToBuffer(double *init_particle_positions,
                                   Omega_h::LO size);

  void MoveToNextLocation(double *particle_origin,
                          double *particle_destinations, int8_t *flying,
                          double *weights, int64_t size);

  void WriteTallyResults();

  void CopyLocationsToBuffer(double *particle_positions) const;

  void MoveToInitialLocation();

  void CopyFlyingFlagToBuffer(int8_t *flying) const;

  void CopyWeightsToBuffer(double *weights) const;

  // --- New: Multi-dimensional tally API methods ---

  /**
   * @brief Register an element-based multi-dimensional tally (call once)
   */
  void AddElementTally(
      const std::vector<uint> &number_of_non_spatial_filter_bins);

  /**
   * @brief Register a node-based (vertex) multi-dimensional tally (call once)
   */
  void AddNodeTally(const std::vector<uint> &number_of_non_spatial_filter_bins);

  /**
   * @brief Set per-particle filter bin assignments for the current step
   * @param bins size = nparticles * n_filters
   */
  void UpdateFilterBins(const std::vector<uint> &bins);

  /**
   * @brief Set the boundary condition for particle-boundary interactions
   */
  void SetReflectiveBoundaryCondition();
};
} // namespace pumitally

#endif // PUMITALLY_PUMITALLYIMPL_H
