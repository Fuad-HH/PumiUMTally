//
// Created by Fuad Hasan on 4/28/26.
//

#ifndef PUMITALLY_PUMITALLYIMPL_H
#define PUMITALLY_PUMITALLYIMPL_H

#include <ParticleTracer.tpp>
#include <pumipic_adjacency.tpp>
#include <pumipic_library.hpp>
#include <pumipic_mesh.hpp>

namespace pumitally {

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

struct ParticleAtElemBoundary {
  /**
   * Allocates tally and other arrays
   * @param num_elements Number of mesh elements
   * @param capacity PUMI-PiC Particle DS capacity
   */
  ParticleAtElemBoundary(Omega_h::LO num_elements, Omega_h::LO capacity);

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
   * Calculate track-length estimated flux
   * @param ptcls PUMI-PiC Particle DS
   * @param xpoints Current intersection points (flat: x0,y0,z0,x1,y1,z1,...)
   * @param elem_ids Current element ID
   * @param ptcl_done If particle tracking is done for this step
   *
   * @details
   * Calculates the tracks segment length inside the current element and
   * it is multiplied with the particle weight before accumulating in the tally
   *
   * @see operator()
   */
  void EvaluateFlux(PPPS *ptcls, const Omega_h::Write<Omega_h::Real> &xpoints,
                    const Omega_h::Write<Omega_h::LO> &elem_ids,
                    const Omega_h::Write<Omega_h::LO> &ptcl_done) const;

  /**
   * Normalize the flux with volume and write it in a vTK file
   * @param full_mesh Omega_h mesh to write the flux result on
   * @param filename VTK file name
   *
   * @see normalizeFlux
   */
  void FinalizeTallies(Omega_h::Mesh &full_mesh,
                       const std::string &filename) const;

  /**
   * Normalize flux with element volumes and number of particles
   * @param mesh Omega_h mesh object
   * @return Omega_h::Reals Normalized flux array
   */
  Omega_h::Reals NormalizeFlux(Omega_h::Mesh &mesh) const;

  /**
   * Mark the tracking step as is_initial_track step
   * @details It turns off tallying for this step
   * @param is_initial If it is initial
   */
  void MarkAsInitial(bool is_initial);

  bool is_initial_track; //!< in is_initial_track run, flux is not tallied
  Omega_h::Write<Omega_h::Real> flux;        //!< Flux tally array
  Omega_h::Write<Omega_h::Real> prev_xpoint; //!< Previous intersection point

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
                          double *weights, Omega_h::LO size);

  void WriteTallyResults();

  void CopyLocationsToBuffer(double *particle_positions) const;

  void MoveToInitialLocation();

  void CopyFlyingFlagToBuffer(int8_t *flying) const;

  void CopyWeightsToBuffer(double *weights) const;
};
} // namespace pumitally

#endif // PUMITALLY_PUMITALLYIMPL_H
