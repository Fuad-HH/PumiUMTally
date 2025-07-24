#ifndef PUMITALLYOPENMC_PUMITALLY_IMPL_TPP
#define PUMITALLYOPENMC_PUMITALLY_IMPL_TPP

// TODO: rename the namespace and even the project to pumitally
// openmc is not necessary in the name since it is general purpose

#include <ParticleTracer.tpp>
#include <pumipic_adjacency.hpp>
#include <pumipic_adjacency.tpp>
#include <pumipic_library.hpp>
#include <pumipic_mesh.hpp>
#include <pumipic_ptcl_ops.hpp>

namespace pumiinopenmc {
struct TallyTimes {
  double initialization_time = 0.0;
  double total_time_to_tally = 0.0;
  double vtk_file_write_time = 0.0;

  void print_times() const;
};

enum class SourceDistribution {
    UNIFORM, // Source uniformly distributed across the mesh
    EQUAL,    // Source at centroids of each element
    ZERO     // in the zeroth element centroid
};

// ------------------------------------------------------------------------------------------------//
// * Data structure for PumiPic
// Particle: 0-origin, 1-destination, 2-particle_id,
// 3-in_advance_particle_queue, 4-weight, 5-group
typedef pumipic::MemberTypes<pumipic::Vector3d, pumipic::Vector3d, Omega_h::LO,
                             Omega_h::I16, Omega_h::Real, int>
    PPParticle;
typedef pumipic::ParticleStructure<PPParticle> PPPS;
typedef Kokkos::DefaultExecutionSpace PPExeSpace;

// ------------------------------------------------------------------------------------------------//
// * Helper Functions
[[deprecated("Use move_to_next_element which is appropriate for new search "
             "class in pumipic. [!Note] It my show this even though it is not "
             "used due to template instantiation.")]]
void pp_move_to_new_element(Omega_h::Mesh &mesh, PPPS *ptcls,
                            Omega_h::Write<Omega_h::LO> &elem_ids,
                            Omega_h::Write<Omega_h::LO> &ptcl_done,
                            Omega_h::Write<Omega_h::LO> &lastExit);

std::unique_ptr<PPPS> pp_create_particle_structure(Omega_h::Mesh mesh,
                                                   pumipic::lid_t numPtcls);

void start_pumi_particles_in_0th_element(Omega_h::Mesh &mesh,
                                         pumiinopenmc::PPPS *ptcls);

// ------------------------------------------------------------------------------------------------//
// * Struct for PumiParticleAtElemBoundary
class PumiParticleAtElemBoundary {
public:
  PumiParticleAtElemBoundary(size_t nelems, size_t capacity);

  void operator()(Omega_h::Mesh &mesh, pumiinopenmc::PPPS *ptcls,
                  Omega_h::Write<Omega_h::LO> &elem_ids,
                  Omega_h::Write<Omega_h::LO> &next_elems,
                  Omega_h::Write<Omega_h::LO> &inter_faces,
                  Omega_h::Write<Omega_h::LO> &lastExit,
                  Omega_h::Write<Omega_h::Real> &inter_points,
                  Omega_h::Write<Omega_h::LO> &ptcl_done,
                  typeof(ptcls->get<0>()) origin_segment,
                  typeof(ptcls->get<1>()) dest_segment);

  void updatePrevXPoint(Omega_h::Write<Omega_h::Real> &xpoints);

  void updatePrevXPoint(PPPS *ptcls);

  void evaluateFlux(PPPS *ptcls, Omega_h::Write<Omega_h::Real> xpoints,
                    Omega_h::Write<Omega_h::LO> elem_ids,
                    Omega_h::Write<Omega_h::LO> ptcl_done);

  void finalizeAndWritePumiFlux(Omega_h::Mesh &full_mesh,
                                const std::string &filename);

  void normalizeFlux(Omega_h::Mesh &mesh);

  void mark_initial_as(bool initial);

  void compute_total_tracklength(PPPS *ptcls);
  void initialize_flux_array(size_t nelems, size_t nEgroups);

  bool initial_; // in initial run, flux is not tallied
  // Dims: nelems, nEgroups, 3(flux, flux^2, standard deviation)
  Kokkos::View<Omega_h::Real ***> flux_;
  Omega_h::Write<Omega_h::Real> prev_xpoint_;
  Omega_h::Write<Omega_h::Real> total_tracklength_;
  Omega_h::Write<int> material_ids_;
};
// ------------------------------------------------------------------------------------------------//

// ------------------------------------------------------------------------------------------------//
// * Struct for PumiTallyImpl
class PumiTallyImpl {
public:
  int64_t pumi_ps_size = 1000000; // hundred thousand
  std::string oh_mesh_fname;

  Omega_h::Library oh_lib;
  Omega_h::Mesh full_mesh_;

  std::unique_ptr<pumipic::Library> pp_lib = nullptr;
  std::unique_ptr<pumipic::Mesh> p_picparts_ = nullptr;
  std::unique_ptr<PPPS> pumipic_ptcls = nullptr;

  long double pumipic_tol = 1e-8;
  bool is_pumipic_initialized = false;
  int64_t iter_count_ = 0;
  double total_initial_weight_ = 0.0;

  std::unique_ptr<PumiParticleAtElemBoundary>
      p_pumi_particle_at_elem_boundary_handler;
  std::unique_ptr<
      ParticleTracer<PPParticle, pumiinopenmc::PumiParticleAtElemBoundary>>
      p_particle_tracer_;

  Omega_h::Write<Omega_h::Real> device_pos_buffer_;
  Omega_h::Write<Omega_h::I8> device_in_adv_que_;
  Omega_h::Write<Omega_h::Real> weights_;
  Omega_h::Write<int> groups_;

  TallyTimes tally_times;

  // * Constructor
  PumiTallyImpl(std::string &mesh_filename, int64_t num_particles, int &argc,
                char **&argv, SourceDistribution source_dist = SourceDistribution::ZERO); // fixme extra &

  // * Destructor
  ~PumiTallyImpl() = default;

  // Functions
  void create_and_initialize_pumi_particle_structure(Omega_h::Mesh *mesh, SourceDistribution source_dist);

  void load_pumipic_mesh_and_init_particles(int &argc, char **&argv,  SourceDistribution source_dist);

  Omega_h::Mesh *partition_pumipic_mesh();

  void init_pumi_libs(int &argc, char **&argv);

  void search_and_rebuild(bool initial, bool migrate = true);

  void read_pumipic_lib_and_full_mesh(int &argc, char **&argv);

  void initialize_particle_location(double *init_particle_positions,
                                    int64_t size);

  void move_to_next_location(double *particle_destinations, int8_t *flying,
                             double *weights, int *groups, int *material_ids,
                             int64_t size);

  void write_pumi_tally_mesh();

  void copy_data_to_device(double *init_particle_positions);

  void search_initial_elements();

  void copy_and_reset_flying_flag(int8_t *flying);

  void copy_weights(double *weights);

  void copy_groups(int *groups);

  void copy_last_location(double *particle_destination, int64_t size);

  void copy_material_ids(int *material_ids, int64_t size);
};

} // namespace pumiinopenmc

#endif // PUMITALLYOPENMC_PUMITALLY_IMPL_TPP
