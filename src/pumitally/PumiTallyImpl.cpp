/**
 * @brief  PumiTallyImpl Implementations
 */

#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>
#include <ParticleTracer.tpp>
#include <pumipic_adjacency.tpp>
#include <pumipic_library.hpp>
#include <pumipic_mesh.hpp>
#include <pumipic_ptcl_ops.hpp>

#include "PumiTallyImpl.h"

namespace pumitally {
std::unique_ptr<PPPS> CreateParticleDS(const Omega_h::Mesh &mesh,
                                       pumipic::lid_t num_ptcls);

void InitializeParticlesInElement0(Omega_h::Mesh &mesh, pumitally::PPPS *ptcls);

void TallyTimes::PrintTimes() const {
  printf("\n");
  printf("[TIME] Initialization time     : %f seconds\n", initialization_time);
  printf("[TIME] Total time to tally     : %f seconds\n", total_time_to_tally);
  printf("[TIME] VTK file write time     : %f seconds\n", vtk_file_write_time);
  printf("[TIME] Total PUMI-Tally time   : %f seconds\n",
         initialization_time + total_time_to_tally + vtk_file_write_time);
}

PumiTallyImpl::PumiTallyImpl(const std::string &mesh_filename,
                             const Omega_h::LO num_ptcls, int argc, char **argv)
    : num_particles(num_ptcls) {
  oh_mesh_filename = mesh_filename;

  position_dev_buffer = Omega_h::Write<Omega_h::Real>(num_particles * 3, 0.0,
                                                      "device_pos_buffer");
  flying_dev_buffer =
      Omega_h::Write<Omega_h::I8>(num_particles, 0, "device_in_adv_que");
  weights_dev_buffer =
      Omega_h::Write<Omega_h::Real>(num_particles, 0.0, "weights");

  // todo can track lengths be here?

  LoadMeshAndInitParticles(argc, argv);
  InitializeParticlesInElement0(*p_picparts->mesh(), pumipic_ptcls.get());

  p_particle_tracer = std::make_unique<
      ParticleTracer<PPParticle, pumitally::ParticleAtElemBoundary>>(
      *p_picparts, pumipic_ptcls.get(),
      *p_pumi_particle_at_elem_boundary_handler, 1e-8);
}

void PumiTallyImpl::CopyInitialPositionToBuffer(double *init_particle_positions,
                                                const Omega_h::LO size) {
  // copy to host buffer
  assert(size == num_particles * 3);
  CopyLocationsToBuffer(init_particle_positions);
  MoveToInitialLocation();
  // TODO Get total is_initial_track particle weight
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::MoveToNextLocation(double *particle_origin,
                                       double *particle_destinations,
                                       int8_t *flying, double *weights,
                                       Omega_h::LO size) {

  // *************** Start Initial Move to Origin ************************** //
  assert(size == num_particles * 3);
  CopyLocationsToBuffer(particle_origin);

  // copy position buffer ps
  auto particle_orig = pumipic_ptcls->get<0>();
  auto particle_dest = pumipic_ptcls->get<1>();
  auto in_flight = pumipic_ptcls->get<3>();
  auto p_wgt = pumipic_ptcls->get<4>();

  // copy fly to device buffer
  CopyFlyingFlagToBuffer(flying);

  const Omega_h::LO pumi_ps_size = num_particles;
  const auto &device_pos_buffer_l = position_dev_buffer;
  const auto &device_in_adv_que_l = flying_dev_buffer;

  auto set_particle_dest_orig =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && pid < pumi_ps_size) {
      // everyone is in flight for this is_initial_track search
      // in_flight(pid) = 1;
      in_flight(pid) = static_cast<unsigned char>(device_in_adv_que_l[pid]);

      if (in_flight(pid) == 1) {
        particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
        particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
        particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];
      } else {
        particle_dest(pid, 0) = particle_orig(pid, 0);
        particle_dest(pid, 1) = particle_orig(pid, 1);
        particle_dest(pid, 2) = particle_orig(pid, 2);
      }

      p_wgt(pid) = 0.0;
    }
  };
  pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest_orig,
                        "set particle orig position as dest");

  bool migrate = iter_count % 100 == 0;
  SearchAndRebuild(/*is_initial_track*/ false, /*migrate*/ true);
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif

  // ************** End Initial Move to Origin ****************************** //

  assert(size == num_particles * 3);

  // copy to device buffer
  CopyLocationsToBuffer(particle_destinations);
  CopyWeightsToBuffer(weights);

  Kokkos::fence();

  auto set_particle_dest =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && pid < num_particles) {
      if (in_flight(pid) == 1) {
        particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
        particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
        particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];
      } else {
        particle_dest(pid, 0) = particle_orig(pid, 0);
        particle_dest(pid, 1) = particle_orig(pid, 1);
        particle_dest(pid, 2) = particle_orig(pid, 2);
      }
    }
  };
  pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest,
                        "set particle position as dest");

  iter_count++;
  SearchAndRebuild(false, migrate);
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::WriteTallyResults() {
  p_pumi_particle_at_elem_boundary_handler->FinalizeTallies(full_mesh,
                                                            "fluxresult.vtk");
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::CopyFlyingFlagToBuffer(int8_t *flying) const {
  // todo get the size too
  const Kokkos::View<Omega_h::I8 *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      flying_host_view(flying, num_particles);
  const Kokkos::View<Omega_h::I8 *, PPExeSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      flying_device_view(flying_dev_buffer.data(), flying_dev_buffer.size());
  Kokkos::deep_copy(flying_device_view, flying_host_view);

  for (int64_t pid = 0; pid < num_particles; ++pid) {
    // reset flying flag to zero // TODO: why, specific reason
    flying[pid] = 0;
  }
}

void PumiTallyImpl::CopyWeightsToBuffer(double *weights) const {
  auto weights_l = weights_dev_buffer;
  Kokkos::View<Omega_h::Real *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_weights_view(weights, num_particles);
  Kokkos::View<Omega_h::Real *, PPExeSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      device_weights_view(weights_l.data(), weights_l.size());

  Kokkos::deep_copy(device_weights_view, host_weights_view);

  auto p_wgt = pumipic_ptcls->get<4>();
  auto copy_particle_weights =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    p_wgt(pid) = weights_l[pid];
  };
  pumipic::parallel_for(pumipic_ptcls.get(), copy_particle_weights,
                        "copy particle weights");
}

void PumiTallyImpl::MoveToInitialLocation() { // assign the location to ptcl
  // dest
  auto particle_dest = pumipic_ptcls->get<1>();
  auto in_flight = pumipic_ptcls->get<3>();

  const int64_t pumi_ps_size_l = num_particles;
  const auto &device_pos_buffer_l = position_dev_buffer;

  auto set_particle_dest =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && pid < pumi_ps_size_l) {
      particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
      particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
      particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];

      // everyone is in flight for this is_initial_track search
      in_flight(pid) = 1;
    }
  };
  pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest,
                        "set is_initial_track position as dest");

  // *is_initial_track* build and search to find the is_initial_track elements
  // of the particles
  SearchAndRebuild(true, true);
  is_pumipic_initialized = true;
}

void PumiTallyImpl::CopyLocationsToBuffer(double *particle_positions) const {
  // fixme it should get size too to avoid memory error
  const Kokkos::View<Omega_h::Real *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      position_host_view(particle_positions,
                         static_cast<size_t>(num_particles * 3));

  const Kokkos::View<Omega_h::Real *, PPExeSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      position_device_view(position_dev_buffer.data(),
                           static_cast<size_t>(num_particles * 3));

  Kokkos::deep_copy(position_device_view, position_host_view);
}

void PumiTallyImpl::InitPUMILibrary(int &argc, char **&argv) {
  pumipic_lib = std::make_unique<pumipic::Library>(&argc, &argv);
  oh_lib = pumipic_lib->omega_h_lib();
}

void UpdateCurrentElement(PPPS *ptcls,
                          const Omega_h::Write<Omega_h::LO> &elem_ids,
                          const Omega_h::Write<Omega_h::LO> &next_elems) {
  const auto in_flight = ptcls->get<3>();
  auto move_to_next = PS_LAMBDA(const int e, const int pid, const int mask) {
    // move only if particle in flight and not leaving the domain
    if (mask > 0 && in_flight(pid) && next_elems[pid] != -1) {
      elem_ids[pid] = next_elems[pid];
    }
  };
  pumipic::parallel_for(ptcls, move_to_next, "move to next element");
}

void ApplyVacuumBC(const Omega_h::Mesh &mesh, PPPS *ptcls,
                   const Omega_h::Write<Omega_h::LO> &elem_ids,
                   const Omega_h::Write<Omega_h::LO> &next_elems,
                   const Omega_h::Write<Omega_h::LO> &ptcl_done,
                   const Omega_h::Write<Omega_h::LO> &last_exit,
                   const Omega_h::Write<Omega_h::LO> &x_face,
                   const Omega_h::Write<Omega_h::Real> &inter_points) {

  // TODO: make this a member variable of the struct
  const auto particle_destination = ptcls->get<1>();
  auto check_exposed_edges =
      PS_LAMBDA(const int e, const int pid, const int mask) {
    if (mask > 0 && !ptcl_done[pid]) {
      const bool reached_destination = (last_exit[pid] == -1);
      const bool hit_boundary =
          ((next_elems[pid] == -1) && (elem_ids[pid] != -1));
      ptcl_done[pid] =
          (reached_destination || hit_boundary) ? 1 : ptcl_done[pid];

      if (hit_boundary) { // just reached the boundary
        x_face[pid] = last_exit[pid];
        // particle reaches the boundary
        particle_destination(pid, 0) = inter_points[pid * 3];
        particle_destination(pid, 1) = inter_points[pid * 3 + 1];
        particle_destination(pid, 2) = inter_points[pid * 3 + 2];
      }
    }
  };
  pumipic::parallel_for(ptcls, check_exposed_edges,
                        "apply vacumm boundary condition");
}

ParticleAtElemBoundary::ParticleAtElemBoundary(const Omega_h::LO num_elements,
                                               const Omega_h::LO capacity)
    : is_initial_track(true), flux(num_elements, 0.0, "flux"),
      prev_xpoint(capacity * 3, 0.0, "prev_xpoint") {
  printf("[INFO] Particle handler at boundary with %d elements and %d "
         "x points size (3 * n_particles)\n",
         flux.size(), prev_xpoint.size());
}

void ParticleAtElemBoundary::operator()(
    const Omega_h::Mesh &mesh, pumitally::PPPS *ptcls,
    const Omega_h::Write<Omega_h::LO> &elem_ids,
    const Omega_h::Write<Omega_h::LO> &next_elems,
    const Omega_h::Write<Omega_h::LO> &inter_faces,
    const Omega_h::Write<Omega_h::LO> &last_exit,
    const Omega_h::Write<Omega_h::Real> &inter_points,
    const Omega_h::Write<Omega_h::LO> &ptcl_done,
    decltype(ptcls->get<0>())
        origin_segment, // NOLINT(performance-unnecessary-value-param)
    decltype(ptcls->get<1>()) dest_segment)
    const { // NOLINT(performance-unnecessary-value-param)
  if (!is_initial_track) {
    EvaluateFlux(ptcls, inter_points, elem_ids, ptcl_done);
    UpdatePreviousXPoints(inter_points);
  }
  ApplyVacuumBC(mesh, ptcls, elem_ids, next_elems, ptcl_done, last_exit,
                inter_faces, inter_points);
  UpdateCurrentElement(ptcls, elem_ids, next_elems);
}

void ParticleAtElemBoundary::MarkAsInitial(const bool is_initial) {
  is_initial_track = is_initial;
}

void ParticleAtElemBoundary::UpdatePreviousXPoints(
    const Omega_h::Write<Omega_h::Real> &xpoints) const {
  OMEGA_H_CHECK_PRINTF(xpoints.size() <= prev_xpoint.size() &&
                           prev_xpoint.size() != 0,
                       "xpoints size %d is greater than prev_xpoint size %d\n",
                       xpoints.size(), prev_xpoint.size());
  const auto &prev_xpoint_l = prev_xpoint;
  auto update = OMEGA_H_LAMBDA(const Omega_h::LO i) {
    prev_xpoint_l[i] = xpoints[i];
  };
  Omega_h::parallel_for(xpoints.size(), update, "update previous xpoints");
}

void ParticleAtElemBoundary::UpdatePreviousXPoints(PPPS *ptcls) const {
  // todo add checks of size
  const auto prev_xpoints_l = prev_xpoint;
  OMEGA_H_CHECK_PRINTF(
      ptcls->capacity() * 3 == prev_xpoints_l.size(),
      "Error: prev_xpoints_s are not size properly capacity %d size %d\n",
      ptcls->capacity(), prev_xpoints_l.size());
  auto xpoints = ptcls->get<0>();
  auto update = PS_LAMBDA(const auto &e, const auto &pid, const auto &mask) {
    prev_xpoints_l[pid * 3 + 0] = xpoints(pid, 0);
    prev_xpoints_l[pid * 3 + 1] = xpoints(pid, 1);
    prev_xpoints_l[pid * 3 + 2] = xpoints(pid, 2);
  };
  pumipic::parallel_for(ptcls, update,
                        "update previous xpoints from origin points");
}

void ParticleAtElemBoundary::EvaluateFlux(
    PPPS *ptcls, const Omega_h::Write<Omega_h::Real> &xpoints,
    const Omega_h::Write<Omega_h::LO> &elem_ids,
    const Omega_h::Write<Omega_h::LO> &ptcl_done) const {
  const auto prev_xpoint_l = prev_xpoint;
  const auto flux_l = flux;
  const auto in_flight = ptcls->get<3>();
  const auto p_wgt = ptcls->get<4>();
  const auto &xpoints_l = xpoints; // todo shouldn't need it, so remove

  auto evaluate_flux =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if ((mask > 0) && (in_flight(pid) == 1) && !ptcl_done[pid]) {
      const Omega_h::Vector<3> dest = {xpoints_l[pid * 3 + 0],
                                       xpoints_l[pid * 3 + 1],
                                       xpoints_l[pid * 3 + 2]};
      const Omega_h::Vector<3> orig = {prev_xpoint_l[pid * 3 + 0],
                                       prev_xpoint_l[pid * 3 + 1],
                                       prev_xpoint_l[pid * 3 + 2]};

      // TODO: Get total number of particles and divide here
      const Omega_h::Real segment_length = Omega_h::norm(dest - orig);

      const Omega_h::Real contribution = segment_length * p_wgt(pid);
      Kokkos::atomic_add(&flux_l[elem_ids[pid]], contribution);
    }
  };
  pumipic::parallel_for(ptcls, evaluate_flux, "flux evaluation loop");
}

Omega_h::Reals
ParticleAtElemBoundary::NormalizeFlux(Omega_h::Mesh &mesh) const {
  const auto &el2n = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;
  const auto &coords = mesh.coords();
  const auto flux_l = flux;

  const Omega_h::Write<Omega_h::Real> tet_volumes(flux.size(), -1.0,
                                                  "tet_volumes");
  const Omega_h::Write<Omega_h::Real> normalized_flux(flux.size(), -1.0,
                                                      "normalized flux");

  auto normalize_flux_with_volume = OMEGA_H_LAMBDA(const Omega_h::LO elem_id) {
    const auto elem_verts = Omega_h::gather_verts<4>(el2n, elem_id);
    const auto elem_vert_coords =
        Omega_h::gather_vectors<4, 3>(coords, elem_verts);

    const auto b = Omega_h::simplex_basis<3, 3>(elem_vert_coords);
    const Omega_h::Real volume = Omega_h::simplex_size_from_basis(b);

    tet_volumes[elem_id] = volume;
    normalized_flux[elem_id] = flux_l[elem_id] / volume;
  };
  Omega_h::parallel_for(tet_volumes.size(), normalize_flux_with_volume,
                        "normalize flux");

  mesh.add_tag(Omega_h::REGION, "volume", 1, Omega_h::Reals(tet_volumes));
  return {normalized_flux};
}

void ParticleAtElemBoundary::FinalizeTallies(
    Omega_h::Mesh &full_mesh, const std::string &filename) const {
  const auto &normalized_flux = NormalizeFlux(full_mesh);
  full_mesh.add_tag(Omega_h::REGION, "flux", 1, normalized_flux);
  Omega_h::vtk::write_parallel(filename, &full_mesh, 3);
}

void CommitParticlePositions(PPPS *ptcls) {
  auto particle_origin = ptcls->get<0>();
  auto particle_destination = ptcls->get<1>();
  auto update_particle_position =
      PS_LAMBDA(const int &, const int &pid, const bool &) {
    particle_origin(pid, 0) = particle_destination(pid, 0);
    particle_origin(pid, 1) = particle_destination(pid, 1);
    particle_origin(pid, 2) = particle_destination(pid, 2);
    particle_destination(pid, 0) = 0.0;
    particle_destination(pid, 1) = 0.0;
    particle_destination(pid, 2) = 0.0;
  };
  ps::parallel_for(ptcls, update_particle_position);
}

void PumiTallyImpl::SearchAndRebuild(const bool initial,
                                     const bool migrate) const {
  // is_initial_track cannot be false when is_pumipic_initialized is false
  // may fail if simulated more than one batch
  assert((is_pumipic_initialized == false && initial == true) ||
         (is_pumipic_initialized == true && initial == false));
  p_pumi_particle_at_elem_boundary_handler->MarkAsInitial(initial);
  auto orig = pumipic_ptcls->get<0>();
  auto dest = pumipic_ptcls->get<1>();
  auto pid = pumipic_ptcls->get<2>();

  if (p_picparts->mesh() == nullptr || p_picparts->mesh()->nelems() == 0) {
    fprintf(stderr, "ERROR: Mesh is empty\n");
  }

  // total tracklengths are used to calculate the flux
  if (!initial) {
    p_pumi_particle_at_elem_boundary_handler->UpdatePreviousXPoints(
        pumipic_ptcls.get());
  }

  const bool found_all = p_particle_tracer->search(migrate);
  if (!found_all) {
    printf(
        "ERROR: Not all particles are found. May need more loops in search\n");
  }
}

std::unique_ptr<PPPS> CreateParticleDS(const Omega_h::Mesh &mesh,
                                       pumipic::lid_t num_ptcls) {
  Omega_h::Int ne = mesh.nelems();
  pumitally::PPPS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  pumitally::PPPS::kkGidView element_gids("element_gids", ne);

  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy;

  Omega_h::parallel_for(
      ne, OMEGA_H_LAMBDA(const Omega_h::LO &i) { element_gids(i) = i; });

  Omega_h::parallel_for(
      mesh.nelems(), OMEGA_H_LAMBDA(const Omega_h::LO id) {
        ptcls_per_elem[id] = (id == 0) ? num_ptcls : 0;
      });

#ifdef PUMI_USE_KOKKOS_CUDA
  printf("PumiPIC Using GPU for simulation...\n");
  policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, 32);
#else
  printf("PumiPIC Using CPU for simulation...\n");
  policy =
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
#endif

  auto ptcls = std::make_unique<pumipic::DPS<pumitally::PPParticle>>(
      policy, ne, num_ptcls, ptcls_per_elem, element_gids);

  return ptcls;
}

void InitializeParticlesInElement0(Omega_h::Mesh &mesh,
                                   pumitally::PPPS *ptcls) {
  // find the centroid of the 0th element
  const auto &coords = mesh.coords();
  const auto &tet2node = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

  Omega_h::Write<Omega_h::Real> centroid_of_el0(3, 0.0, "centroid");

  auto find_centroid_of_el0 = OMEGA_H_LAMBDA(const Omega_h::LO id) {
    const auto nodes = Omega_h::gather_verts<4>(tet2node, id);
    const Omega_h::Few<Omega_h::Vector<3>, 4> tet_node_coords =
        Omega_h::gather_vectors<4, 3>(coords, nodes);
    const auto centroid = o::average(tet_node_coords);
    centroid_of_el0[0] = centroid[0];
    centroid_of_el0[1] = centroid[1];
    centroid_of_el0[2] = centroid[2];
  };
  Omega_h::parallel_for(1, find_centroid_of_el0, "find centroid of element 0");

  // assign the location to all particles
  auto init_loc = ptcls->get<0>();
  auto pids = ptcls->get<2>();
  auto in_fly = ptcls->get<3>();

  auto set_initial_positions =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0) {
      pids(pid) = pid;
      in_fly(pid) = 1;
      init_loc(pid, 0) = centroid_of_el0[0];
      init_loc(pid, 1) = centroid_of_el0[1];
      init_loc(pid, 2) = centroid_of_el0[2];
    }
  };
  pumipic::parallel_for(ptcls, set_initial_positions,
                        "set is_initial_track particle positions");
}

Omega_h::Mesh PumiTallyImpl::PartitionMesh() {
  const Omega_h::Write<Omega_h::LO> owners(full_mesh.nelems(), 0, "owners");
  // all the particles are initialized in element 0 to do an is_initial_track
  // search to find the starting locations of the openmc given particles.
  // p_picparts = new pumipic::Mesh(full_mesh, Omega_h::LOs(owners));
  p_picparts = std::make_unique<pumipic::Mesh>(full_mesh, Omega_h::LOs(owners));
  printf("PumiPIC mesh partitioned\n");

  return *p_picparts->mesh();
}

void PumiTallyImpl::InitializePUMIParticleStructure(Omega_h::Mesh &mesh) {
  pumipic_ptcls = CreateParticleDS(mesh, num_particles);
  InitializeParticlesInElement0(mesh, pumipic_ptcls.get());
  p_pumi_particle_at_elem_boundary_handler =
      std::make_unique<pumitally::ParticleAtElemBoundary>(
          mesh.nelems(), pumipic_ptcls->capacity());

  printf("PumiPIC Mesh and data structure created with %d and %d as particle "
         "structure capacity\n",
         p_picparts->mesh()->nelems(), pumipic_ptcls->capacity());
}

void PumiTallyImpl::ReadFullMesh(int &argc, char **&argv) {
  printf("Reading the Omega_h mesh %s to tally with tracklength estimator\n",
         oh_mesh_filename.c_str());
  InitPUMILibrary(argc, argv);

  if (oh_mesh_filename.empty()) {
    printf("[ERROR] Omega_h mesh for PumiPIC is not given. Provide --ohMesh = "
           "<osh file>");
  }
  full_mesh = Omega_h::binary::read(oh_mesh_filename, &oh_lib);
  if (full_mesh.dim() != 3) {
    printf("PumiPIC only works for 3D mesh now.\n");
  }
  printf("PumiPIC Loaded mesh %s with %d elements\n", oh_mesh_filename.c_str(),
         full_mesh.nelems());
}

void PumiTallyImpl::LoadMeshAndInitParticles(int &argc, char **&argv) {
  ReadFullMesh(argc, argv);
  Omega_h::Mesh mesh = PartitionMesh();
  InitializePUMIParticleStructure(mesh);
}
} // namespace pumitally
