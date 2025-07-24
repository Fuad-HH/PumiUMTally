/*
 * This is created as a proof of concept for the Degas2 Neutral Transport Code
 * Provides parallel search capabilities and tallying to degas2 through this
 * PUMI-Tally interface
 */

#include <iostream>
#include <stdexcept>
#include <vector>

#include "DG2Physics.h"
#include "pumitally_impl.tpp"

// ******************************************* Helper Functions
// ******************************************* //
enum class SourceDistribution {
  UNIFORM, // Source uniformly distributed across the mesh
  EQUAL    // Source at centroids of each element
};
struct InputParameters {
  std::string mesh_name;
  int num_particles = 0;
  int max_iterations = 1000;
  SourceDistribution source_distribution = SourceDistribution::EQUAL;
};
void read_input_parameters(int argc, char *const *argv,
                           InputParameters &params);
void print_initial_info(const std::string &mesh_name,
                        int num_particles) noexcept;
// computes centroids of each element in the mesh and stores them in a tag
void get_centroids(Omega_h::Mesh &mesh,
                   Omega_h::Write<Omega_h::Real> centroids);

template <typename T>
void vector2write(const std::vector<T> &vec, Omega_h::Write<T> &write_vec) {
  Omega_h::HostWrite<T> host_write_vec(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    host_write_vec[i] = vec[i];
  }
  write_vec = Omega_h::Write<T>(host_write_vec);
}

struct Fields {
  std::vector<double> electron_temperature;
  std::vector<double> ion_temperature;
  std::vector<double> electron_density;
  std::vector<double> ion_density;
  std::vector<double> bulk_flow_velocity;
};
void get_field_values(Omega_h::Reals centroids, Fields &fields);
void set_field_values_to_mesh(Omega_h::Mesh &mesh, const Fields &fields);
void set_source_particles(Omega_h::Mesh &mesh,
                          SourceDistribution source_distribution,
                          Omega_h::Write<Omega_h::Real> particle_positions);
void transport(pumiinopenmc::PumiTallyImpl &pumi_tally, DG2Physics &physics,
               const Omega_h::Read<Omega_h::Real> &electron_density,
               const Omega_h::Read<Omega_h::Real> ion_density,
               const Omega_h::Read<Omega_h::Real> electron_temperature,
               const Omega_h::Read<Omega_h::Real> ion_temperature,
               const Omega_h::Read<Omega_h::Real> bulk_flow_velocity,
               int max_iterations);

// ***************************************************************************//
// *************************** Main Function ******************************** //
// ***************************************************************************//
int main(int argc, char *argv[]) {

  // Read input parameters
  InputParameters input_params;
  read_input_parameters(argc, argv, input_params);
  print_initial_info(input_params.mesh_name, input_params.num_particles);

  // Initialize PUMI-Tally and Read Fields
  auto pumi_tally = pumiinopenmc::PumiTallyImpl(
      input_params.mesh_name, input_params.num_particles, argc, argv);
  auto &mesh = pumi_tally.full_mesh_;
  Omega_h::Write<Omega_h::Real> centroids(mesh.nelems() * 3);
  get_centroids(mesh, centroids);
  Fields fields;
  get_field_values(Omega_h::Reals(centroids), fields);
  set_field_values_to_mesh(mesh, fields);

  Kokkos::View<Omega_h::Real ***> sigma_t_;            // mat, T, g
  Kokkos::View<Omega_h::Real ***> sigma_a_;            // mat, T, g
  Kokkos::View<Omega_h::Real ****> scattering_matrix_; // mat, T, g, g
  Kokkos::View<Omega_h::Real ***> sigma_s_;            // mat, T, g
  DG2CrossSection crossSection(sigma_t_, sigma_a_, scattering_matrix_,
                               sigma_s_);
  DG2Physics physics(crossSection, input_params.num_particles);

  // Initialize particles
  Omega_h::Write<Omega_h::Real> particle_positions(pumi_tally.pumi_ps_size * 3,
                                                   "particle_positions");
  set_source_particles(mesh, input_params.source_distribution,
                       particle_positions);
  pumi_tally.device_pos_buffer_ = particle_positions;
  pumi_tally.search_initial_elements();

  // Move particles
  auto electron_density =
      mesh.get_array<Omega_h::Real>(Omega_h::REGION, "electron_density");
  auto ion_density =
      mesh.get_array<Omega_h::Real>(Omega_h::REGION, "ion_density");
  auto electron_temperature =
      mesh.get_array<Omega_h::Real>(Omega_h::REGION, "electron_temperature");
  auto ion_temperature =
      mesh.get_array<Omega_h::Real>(Omega_h::REGION, "ion_temperature");
  auto bulk_flow_velocity =
      mesh.get_array<Omega_h::Real>(Omega_h::REGION, "bulk_flow_velocity");

  transport(pumi_tally, physics, electron_density, ion_density,
            electron_temperature, ion_temperature, bulk_flow_velocity,
            input_params.max_iterations);

  // Finalize and output results
  pumi_tally.write_pumi_tally_mesh();

  return 0;
}

// ****************************************************************************************************************
// //
// ********************************************* Function Definitions
// ********************************************* //
// ****************************************************************************************************************
// //

void transport(pumiinopenmc::PumiTallyImpl &pumi_tally, DG2Physics &physics,
               const Omega_h::Read<Omega_h::Real> &electron_density,
               const Omega_h::Read<Omega_h::Real> ion_density,
               const Omega_h::Read<Omega_h::Real> electron_temperature,
               const Omega_h::Read<Omega_h::Real> ion_temperature,
               const Omega_h::Read<Omega_h::Real> bulk_flow_velocity,
               int max_iterations) {

  for (int iter = 0; iter < max_iterations; ++iter) {
    auto particle_dest = pumi_tally.pumipic_ptcls->get<1>();
    auto particle_orig = pumi_tally.pumipic_ptcls->get<0>();
    auto particle_weight = pumi_tally.pumipic_ptcls->get<4>();
    auto particle_group = pumi_tally.pumipic_ptcls->get<5>();
    auto get_new_destination =
        PS_LAMBDA(const int &e, const int &pid, const int &mask) {
      if (mask > 0) { // FIXME: check if the particle is at destination or at
                      // the boundary
        ParticleInfo particle_info;
        particle_info.position[0] = particle_dest(pid, 0);
        particle_info.position[1] = particle_dest(pid, 1);
        particle_info.position[2] = particle_dest(pid, 2);
        auto direction = Omega_h::normalize(Omega_h::Vector<3>{
            particle_info.position[0] - particle_orig(pid, 0),
            particle_info.position[1] - particle_orig(pid, 1),
            particle_info.position[2] - particle_orig(pid, 2)});
        particle_info.direction[0] = direction[0];
        particle_info.direction[1] = direction[1];
        particle_info.direction[2] = direction[2];
        particle_info.weight = particle_weight(pid);
        particle_info.energy_group = particle_group(pid);
        particle_info.particle_index = pid;

        FieldInfo field_info;
        field_info.electron_temperature = electron_temperature[e];
        field_info.ion_temperature = ion_temperature[e];
        field_info.electron_density = electron_density[e];
        field_info.ion_density = ion_density[e];
        field_info.bulk_flow_velocity[0] = bulk_flow_velocity[e * 3 + 0];
        field_info.bulk_flow_velocity[1] = bulk_flow_velocity[e * 3 + 1];
        field_info.bulk_flow_velocity[2] = bulk_flow_velocity[e * 3 + 2];

        physics.collide_particle(particle_info, field_info);
        physics.sample_collision_distance(particle_info, field_info);

        // Update particle position and direction
        particle_dest(pid, 0) = particle_info.position[0];
        particle_dest(pid, 1) = particle_info.position[1];
        particle_dest(pid, 2) = particle_info.position[2];
        particle_weight(pid) = particle_info.weight;
        particle_group(pid) = particle_info.energy_group;
      }
    };
    pumipic::parallel_for(pumi_tally.pumipic_ptcls.get(), get_new_destination,
                          "get new destination");

    pumi_tally.search_and_rebuild(
        false, true); // for now, always rebuild the pp structure
    Kokkos::fence();
  }
}

void initialize_equal_source(const Omega_h::Mesh &mesh,
                             Omega_h::Write<Omega_h::Real> particle_positions) {
  Omega_h::LO num_particles = particle_positions.size() / 3;
  Omega_h::LO particles_per_element = num_particles / mesh.nelems();
  Omega_h::LO remainder_particles = num_particles % mesh.nelems();
  const auto centroids =
      mesh.get_array<Omega_h::Real>(Omega_h::REGION, "centroid");

  // set the particle positions to the centroids of each element,
  // and adding one particle to the first few elements if there are remainder
  // particles
  Omega_h::parallel_for(
      "set source particles", mesh.nelems(), OMEGA_H_LAMBDA(Omega_h::LO e) {
        Omega_h::Vector<3> centroid = {
            centroids[e * 3 + 0], centroids[e * 3 + 1], centroids[e * 3 + 2]};
        Omega_h::LO particles_in_this_element =
            particles_per_element + (e < remainder_particles ? 1 : 0);
        for (Omega_h::LO p = 0; p < particles_in_this_element; ++p) {
          Omega_h::LO particle_index =
              e * particles_per_element + p +
              (e < remainder_particles ? e : remainder_particles);
          particle_positions[particle_index * 3 + 0] = centroid[0];
          particle_positions[particle_index * 3 + 1] = centroid[1];
          particle_positions[particle_index * 3 + 2] = centroid[2];
        }
      });
  Kokkos::fence();
}

void set_source_particles(Omega_h::Mesh &mesh,
                          const SourceDistribution source_distribution,
                          Omega_h::Write<Omega_h::Real> particle_positions) {
  OMEGA_H_CHECK_PRINTF(particle_positions.size() > 0,
                       "Particle positions size (%ld) must be greater than 0\n",
                       particle_positions.size());

  switch (source_distribution) {
  case SourceDistribution::UNIFORM:
    throw std::runtime_error("Uniform distribution is not implemented yet.");
    break;
  case SourceDistribution::EQUAL: {
    initialize_equal_source(mesh, particle_positions);

    break;
  }
  default:
    throw std::runtime_error("Unsupported source distribution!\n");
  }
}

void set_field_values_to_mesh(Omega_h::Mesh &mesh, const Fields &fields) {
  OMEGA_H_CHECK_PRINTF(fields.electron_temperature.size() == mesh.nelems(),
                       "Electron temperature size (%d) must be equal to number "
                       "of elements (%d)\n",
                       fields.electron_temperature.size(), mesh.nelems());
  OMEGA_H_CHECK_PRINTF(
      fields.ion_temperature.size() == mesh.nelems(),
      "Ion temperature size (%d) must be equal to number of elements (%d)\n",
      fields.ion_temperature.size(), mesh.nelems());
  OMEGA_H_CHECK_PRINTF(
      fields.electron_density.size() == mesh.nelems(),
      "Electron density size (%d) must be equal to number of elements (%d)\n",
      fields.electron_density.size(), mesh.nelems());
  OMEGA_H_CHECK_PRINTF(
      fields.ion_density.size() == mesh.nelems(),
      "Ion density size (%d) must be equal to number of elements (%d)\n",
      fields.ion_density.size(), mesh.nelems());
  OMEGA_H_CHECK_PRINTF(fields.bulk_flow_velocity.size() == mesh.nelems() * 3,
                       "Bulk flow velocity size (%d) must be equal to number "
                       "of elements * 3 (%d)\n",
                       fields.bulk_flow_velocity.size(), mesh.nelems() * 3);

  Omega_h::Write<Omega_h::Real> electron_temperature(
      fields.electron_temperature.size());
  Omega_h::Write<Omega_h::Real> ion_temperature(fields.ion_temperature.size());
  Omega_h::Write<Omega_h::Real> electron_density(
      fields.electron_density.size());
  Omega_h::Write<Omega_h::Real> ion_density(fields.ion_density.size());
  Omega_h::Write<Omega_h::Real> bulk_flow_velocity(
      fields.bulk_flow_velocity.size());
  vector2write(fields.electron_temperature, electron_temperature);
  vector2write(fields.ion_temperature, ion_temperature);
  vector2write(fields.electron_density, electron_density);
  vector2write(fields.ion_density, ion_density);
  vector2write(fields.bulk_flow_velocity, bulk_flow_velocity);

  // Assuming the mesh has a field for each of the properties
  mesh.add_tag<Omega_h::Real>(Omega_h::REGION, "electron_temperature", 1,
                              electron_temperature);
  mesh.add_tag<Omega_h::Real>(Omega_h::REGION, "ion_temperature", 1,
                              ion_temperature);
  mesh.add_tag<Omega_h::Real>(Omega_h::REGION, "electron_density", 1,
                              electron_density);
  mesh.add_tag<Omega_h::Real>(Omega_h::REGION, "ion_density", 1, ion_density);
  mesh.add_tag<Omega_h::Real>(Omega_h::REGION, "bulk_flow_velocity", 3,
                              bulk_flow_velocity);
}

void get_centroids(Omega_h::Mesh &mesh,
                   Omega_h::Write<Omega_h::Real> centroids) {
  OMEGA_H_CHECK_PRINTF(
      centroids.size() == mesh.nelems() * 3,
      "Centroids size (%d) must be equal to number of elements * 3 (%d)\n",
      centroids.size(), mesh.nelems() * 3);
  const auto coords = mesh.coords();
  const auto nelems = mesh.nelems();
  const auto e2v = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

  // FIXME: Hardcoded for 3D tets
  Omega_h::parallel_for(
      "calculate centroids", nelems, OMEGA_H_LAMBDA(int e) {
        auto nodes = o::gather_verts<4>(e2v, e);
        o::Few<o::Vector<3>, 4> elem_coords =
            o::gather_vectors<4, 3>(coords, nodes);
        o::Vector<3> centroid = o::average(elem_coords);

        centroids[e * 3 + 0] = centroid[0];
        centroids[e * 3 + 1] = centroid[1];
        centroids[e * 3 + 2] = centroid[2];
      });

  mesh.add_tag<Omega_h::Real>(Omega_h::REGION, "centroid", 3, centroids);
}

// TODO: Implement the function to retrieve field values based on centroids
void get_field_values(Omega_h::Reals centroids, Fields &fields) {
  auto centroids_h = Omega_h::HostRead<Omega_h::Real>(centroids);

  // This function should be implemented to retrieve the field values
  // based on the centroids provided (read centroids_h on CPU.
  // It contains centroids like e0_0, e0_1, e0_2, e1_0, e1_1, e1_2, ....)
  // For now, we will just fill the fields with dummy data.
  size_t num_elements = centroids.size() / 3; // Assuming 3D
  fields.electron_temperature.resize(num_elements, 1.0);
  fields.ion_temperature.resize(num_elements, 1.0);
  fields.electron_density.resize(num_elements, 1.0);
  fields.ion_density.resize(num_elements, 1.0);
  fields.bulk_flow_velocity.resize(num_elements, 0.0);

  // TODO: Add your code here to retrieve the actual field values
}

void print_initial_info(const std::string &mesh_name,
                        int num_particles) noexcept {
  printf("\n===================> Accelerated Degas2 <======================\n");
  std::string mesh_particle_info =
      "\tMesh: " + mesh_name +
      "\n\tNumber of particles: " + std::to_string(num_particles) + "\n";
  printf("%s", mesh_particle_info.c_str());
  printf("\n===============================================================\n");
}

void read_input_parameters(int argc, char *const *argv,
                           InputParameters &params) {
  if (argc != 5) {
    throw std::runtime_error(
        "Usage: " + std::string(argv[0]) +
        " <mesh_name> <num_particles> <max iter> <source_distribution>");
  }
  params.mesh_name = argv[1];
  params.num_particles = std::stoi(argv[2]);
  if (params.num_particles <= 0) {
    throw std::runtime_error("Number of particles must be a positive integer.");
  }
  params.max_iterations = std::stoi(argv[3]);

  std::string source_dist_str = argv[4];
  // FIXME: hardcoded lower case
  if (source_dist_str == "uniform") {
    params.source_distribution = SourceDistribution::UNIFORM;
  } else if (source_dist_str == "equal") {
    params.source_distribution = SourceDistribution::EQUAL;
  } else {
    throw std::runtime_error(
        "Invalid source distribution. Use 'uniform' or 'equal'.");
  }
}
