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
                                       pumipic::lid_t num_ptcls,
                                       std::vector<Omega_h::LO> *out_ptcls_per_elem = nullptr);

void InitializeParticlesInElement0(Omega_h::Mesh &mesh, pumitally::PPPS *ptcls);
void InitializeParticlesInRegion(Omega_h::Mesh &mesh, pumitally::PPPS *ptcls,
                                 int region_id);

Omega_h::Reals GetCentroids(Omega_h::Mesh &mesh, const bool add_tag) {
  const auto coords = mesh.coords();
  const auto nelems = mesh.nelems();
  const auto e2v = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

  const Omega_h::Write<Omega_h::Real> centroids(nelems * 3, 0.0, "centroids");

  // FIXME: Hardcoded for 3D tets
  Omega_h::parallel_for(
      "calculate centroids", nelems, OMEGA_H_LAMBDA(int e) {
        const auto nodes = o::gather_verts<4>(e2v, e);
        o::Few<o::Vector<3>, 4> elem_coords =
            o::gather_vectors<4, 3>(coords, nodes);
        o::Vector<3> centroid = o::average(elem_coords);

        centroids[e * 3 + 0] = centroid[0];
        centroids[e * 3 + 1] = centroid[1];
        centroids[e * 3 + 2] = centroid[2];
      });

  if (add_tag) {
    mesh.add_tag<Omega_h::Real>(Omega_h::REGION, "centroid", 3, centroids);
  }

  return centroids;
}

void TallyTimes::PrintTimes() const {
  printf("\n");
  printf("[TIME] Initialization time     : %f seconds\n", initialization_time);
  printf("[TIME] Total time to tally     : %f seconds\n", total_time_to_tally);
  printf("[TIME] VTK file write time     : %f seconds\n", vtk_file_write_time);
  printf("[TIME] Total PUMI-Tally time   : %f seconds\n",
         initialization_time + total_time_to_tally + vtk_file_write_time);
}

PumiTallyImpl::PumiTallyImpl(const std::string &mesh_filename,
                             const Omega_h::LO num_ptcls, int argc, char **argv,
                             const SourceDistribution source_dist,
                             int source_region_id)
    : num_particles(num_ptcls), source_region_id(source_region_id) {
  oh_mesh_filename = mesh_filename;

  position_dev_buffer = Omega_h::Write<Omega_h::Real>(num_particles * 3, 0.0,
                                                      "device_pos_buffer");
  flying_dev_buffer =
      Omega_h::Write<Omega_h::I8>(num_particles, 0, "device_in_adv_que");
  weights_dev_buffer =
      Omega_h::Write<Omega_h::Real>(num_particles, 0.0, "weights");

  // todo can track lengths be here?

  // Read and partition the mesh (common to all source types)
  ReadFullMesh(argc, argv);
  Omega_h::Mesh mesh = PartitionMesh();

  // Initialize particle structure based on source type
  if (source_dist == SourceDistribution::REGION) {
    InitializePUMIParticleStructureForRegion(mesh, source_region_id);
  } else {
    InitializePUMIParticleStructure(mesh);
  }

  switch (source_dist) {
  case SourceDistribution::UNIFORM:
    throw std::runtime_error(
        "UNIFORM source distribution is not implemented yet");
    break;
  case SourceDistribution::EQUAL:
    throw std::runtime_error(
        "EQUAL source distribution is not implemented yet");
    break;
  case SourceDistribution::ZERO:
    InitializeParticlesInElement0(*p_picparts->mesh(), pumipic_ptcls.get());
    break;
  case SourceDistribution::REGION:
    InitializeParticlesInRegion(*p_picparts->mesh(), pumipic_ptcls.get(),
                                source_region_id);
    break;
  default:
    throw std::runtime_error("Invalid source distribution");
  }

  p_particle_tracer = std::make_unique<
      ParticleTracer<PPParticle, pumitally::ParticleAtElemBoundary>>(
      *p_picparts, pumipic_ptcls.get(),
      *p_pumi_particle_at_elem_boundary_handler, 1e-8);
}

void PumiTallyImpl::CopyInitialPositionToBuffer(double *init_particle_positions,
                                                const Omega_h::LO size) {
  assert(size == num_particles * 3);
  CopyLocationsToBuffer(init_particle_positions);
  MoveToInitialLocation();
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::MoveToNextLocation(double *particle_origin,
                                       double *particle_destinations,
                                       int8_t *flying, double *weights,
                                       int64_t size) {

  // *************** Start Initial Move to Origin ************************** //
  assert(size == num_particles * 3);
  CopyLocationsToBuffer(particle_origin);

  auto particle_orig = pumipic_ptcls->get<0>();
  auto particle_dest = pumipic_ptcls->get<1>();
  auto in_flight = pumipic_ptcls->get<3>();
  auto p_wgt = pumipic_ptcls->get<4>();

  CopyFlyingFlagToBuffer(flying);

  const Omega_h::LO pumi_ps_size = num_particles;
  const auto &device_pos_buffer_l = position_dev_buffer;
  const auto &device_in_adv_que_l = flying_dev_buffer;

  auto set_particle_dest_orig =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && pid < pumi_ps_size) {
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
  p_pumi_particle_at_elem_boundary_handler->FinalizeTallies(
      full_mesh, "fluxresult.vtk", source_ptcls_per_elem);
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::CopyFlyingFlagToBuffer(int8_t *flying) const {
  const Kokkos::View<Omega_h::I8 *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      flying_host_view(flying, num_particles);
  const Kokkos::View<Omega_h::I8 *, PPExeSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      flying_device_view(flying_dev_buffer.data(), flying_dev_buffer.size());
  Kokkos::deep_copy(flying_device_view, flying_host_view);

  for (int64_t pid = 0; pid < num_particles; ++pid) {
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

void PumiTallyImpl::MoveToInitialLocation() {
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

      in_flight(pid) = 1;
    }
  };
  pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest,
                        "set is_initial_track position as dest");

  SearchAndRebuild(true, true);
  is_pumipic_initialized = true;
}

void PumiTallyImpl::CopyLocationsToBuffer(double *particle_positions) const {
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

// ==========================================================================
// PumiTallyImpl: Tally registration and filter bin API
// ==========================================================================

void PumiTallyImpl::AddElementTally(
    const std::vector<uint> &number_of_non_spatial_filter_bins) {
  p_pumi_particle_at_elem_boundary_handler->AddElementTally(
      number_of_non_spatial_filter_bins);
}

void PumiTallyImpl::AddNodeTally(
    const std::vector<uint> &number_of_non_spatial_filter_bins) {
  p_pumi_particle_at_elem_boundary_handler->AddNodeTally(
      number_of_non_spatial_filter_bins);
}

void PumiTallyImpl::UpdateFilterBins(const std::vector<uint> &bins) {
  const auto &spec = p_pumi_particle_at_elem_boundary_handler->GetTallySpec();
  OMEGA_H_CHECK(spec.is_initialized);

  size_t expected_size =
      static_cast<size_t>(num_particles) * spec.GetNumFilters();
  OMEGA_H_CHECK_PRINTF(
      bins.size() == expected_size,
      "Filter bins size (%zu) must equal num_particles (%d) * n_filters "
      "(%u) = %zu\n",
      bins.size(), num_particles, spec.GetNumFilters(), expected_size);

  p_pumi_particle_at_elem_boundary_handler->SetFilterBins(bins);
}

void PumiTallyImpl::UpdateFilterBins(const Kokkos::View<uint *> &bins) {
  const auto &spec = p_pumi_particle_at_elem_boundary_handler->GetTallySpec();
  OMEGA_H_CHECK(spec.is_initialized);

  size_t expected_size =
      static_cast<size_t>(num_particles) * spec.GetNumFilters();
  OMEGA_H_CHECK_PRINTF(
      static_cast<size_t>(bins.size()) == expected_size,
      "Filter bins View size (%d) must equal num_particles (%d) * n_filters "
      "(%u) = %zu\n",
      bins.size(), num_particles, spec.GetNumFilters(), expected_size);

  p_pumi_particle_at_elem_boundary_handler->SetFilterBins(bins);
}

void PumiTallyImpl::SetReflectiveBoundaryCondition() {
  p_pumi_particle_at_elem_boundary_handler->SetBoundaryCondition(
      ParticleAtElemBoundary::BoundaryCondition::REFLECTIVE,
      *p_picparts->mesh());
}

// ==========================================================================
// Free functions: element update, normals, volume, BC
// ==========================================================================

void UpdateCurrentElement(PPPS *ptcls,
                          const Omega_h::Write<Omega_h::LO> &elem_ids,
                          const Omega_h::Write<Omega_h::LO> &next_elems) {
  const auto in_flight = ptcls->get<3>();
  auto move_to_next = PS_LAMBDA(const int e, const int pid, const int mask) {
    if (mask > 0 && in_flight(pid) && next_elems[pid] != -1) {
      elem_ids[pid] = next_elems[pid];
    }
  };
  pumipic::parallel_for(ptcls, move_to_next, "move to next element");
}

void compute_boundary_normals(Omega_h::Mesh &mesh) {
  const auto exposed_edges = Omega_h::mark_exposed_sides(&mesh);
  const auto face2elems = mesh.ask_up(mesh.dim() - 1, mesh.dim()).ab2b;
  const auto face2elemsOffset = mesh.ask_up(mesh.dim() - 1, mesh.dim()).a2ab;
  const auto elem2nodes = mesh.ask_down(mesh.dim(), 0).ab2b;
  const auto face2nodes = mesh.ask_down(mesh.dim() - 1, 0).ab2b;
  const auto coords = mesh.coords();

  Omega_h::Write<Omega_h::Real> normals(mesh.nfaces() * 3, 0.0,
                                        "boundary_normals");

  auto calculate_normals = OMEGA_H_LAMBDA(const Omega_h::LO &face_id) {
    if (exposed_edges[face_id]) {
      const auto face_nodes = Omega_h::gather_verts<3>(face2nodes, face_id);
      const auto face_coords =
          Omega_h::gather_vectors<3, 3>(coords, face_nodes);
      const auto normal = Omega_h::cross(
          face_coords[1] - face_coords[0],
          face_coords[2] - face_coords[0]);

      const auto norm = Omega_h::norm(normal);

      const auto elem_id = face2elems[face2elemsOffset[face_id]];
      const auto elem_nodes = Omega_h::gather_verts<4>(elem2nodes, elem_id);
      int fourth_node = -1;
      for (int i = 0; i < 4; ++i) {
        if (elem_nodes[i] != face_nodes[0] && elem_nodes[i] != face_nodes[1] &&
            elem_nodes[i] != face_nodes[2]) {
          fourth_node = elem_nodes[i];
          break;
        }
      }
      OMEGA_H_CHECK_PRINTF(fourth_node != -1,
                           "Error: fourth node not found for face %d\n",
                           face_id);

      const Omega_h::Vector<3> fourth_node_coord = {
          coords[fourth_node * 3 + 0], coords[fourth_node * 3 + 1],
          coords[fourth_node * 3 + 2]};

      Omega_h::Vector<3> fourth_2_face_vector = {
          fourth_node_coord[0] - face_coords[0][0],
          fourth_node_coord[1] - face_coords[0][1],
          fourth_node_coord[2] - face_coords[0][2]};

      Omega_h::Vector<3> inner_norm;
      if (Omega_h::inner_product(normal, fourth_2_face_vector) < 0) {
        for (int i = 0; i < 3; ++i) {
          inner_norm[i] = -normal[i] / norm;
        }
      } else {
        for (int i = 0; i < 3; ++i) {
          inner_norm[i] = normal[i] / norm;
        }
      }
      normals[face_id * 3 + 0] = inner_norm[0];
      normals[face_id * 3 + 1] = inner_norm[1];
      normals[face_id * 3 + 2] = inner_norm[2];
    }
  };
  Omega_h::parallel_for(mesh.nfaces(), calculate_normals,
                        "compute boundary normals");
  mesh.add_tag(Omega_h::FACE, "normals", 3, Omega_h::Reals(normals));
}

OMEGA_H_DEVICE o::Real volume_tet(const o::Few<o::Vector<3>, 4> &tet_verts) {
  o::Few<o::Vector<3>, 3> basis33 = {tet_verts[1] - tet_verts[0],
                                     tet_verts[2] - tet_verts[0],
                                     tet_verts[3] - tet_verts[0]};
  auto volume = o::tet_volume_from_basis(basis33);
  return volume;
}

o::Real volume_of_3d_mesh(o::Mesh &mesh) {
  OMEGA_H_CHECK_PRINTF(mesh.dim() == 3,
                       "Volume calculation is only supported for 3D meshes, "
                       "but got %dD mesh\n",
                       mesh.dim());
  const auto coords = mesh.coords();
  const auto elems2nodes = mesh.ask_down(o::REGION, o::VERT).ab2b;
  const auto n_elems = mesh.nelems();
  o::Real total_volume = 0.0;

  Kokkos::parallel_reduce(
      n_elems,
      KOKKOS_LAMBDA(const int i, o::Real &local_volume) {
        auto elem_nodes = o::gather_verts<4>(elems2nodes, i);
        o::Few<o::Vector<3>, 4> elem_coords;
        elem_coords = o::gather_vectors<4, 3>(coords, elem_nodes);
        o::Real elem_volume = volume_tet(elem_coords);
        local_volume += elem_volume;
      },
      Kokkos::Sum<o::Real>(total_volume));

  return total_volume;
}

// OpenMC uses continuous reflection approach,
// that is, the particle direction changes and
// then a new path length is sampled.
void apply_reflection_boundary_condition(
    const Omega_h::Mesh &mesh, PPPS *ptcls,
    const Omega_h::Write<Omega_h::LO> &elem_ids,
    const Omega_h::Write<Omega_h::LO> &next_elems,
    const Omega_h::Write<Omega_h::LO> &ptcl_done,
    const Omega_h::Write<Omega_h::LO> &lastExit,
    const Omega_h::Write<Omega_h::LO> &xFace,
    const Omega_h::Write<Omega_h::Real> &inter_points,
    const Omega_h::Write<int> &material_ids, bool initial,
    const Omega_h::Reals &face_normals) {

  auto particle_destination = ptcls->get<1>();
  auto particle_origin = ptcls->get<0>();

  const auto class_ids = mesh.get_array<int>(3, "class_id");
  const auto normals = face_normals;

  auto checkExposedEdges =
      PS_LAMBDA(const int e, const int pid, const int mask) {
    if (mask > 0 && !ptcl_done[pid]) {
      bool reached_destination = (lastExit[pid] == -1);
      bool hit_outer_boundary =
          ((next_elems[pid] == -1) && (elem_ids[pid] != -1));

      bool hit_material_boundary = false;
      if (!initial) {
        if (next_elems[pid] != -1) {
          if (class_ids[elem_ids[pid]] != class_ids[next_elems[pid]]) {
            hit_material_boundary = true;
            material_ids[pid] = class_ids[next_elems[pid]];
          }
        } else {
          material_ids[pid] = -1;
        }
      }

      // Material boundary: stop the particle at the intersection point.
      // Keep it in the CURRENT element (where the point is on a face, so
      // the barycentric check passes cleanly).  Signal to Transport via
      // lastExit = -2 that this was a material boundary (not a destination
      // and not a domain boundary), so Transport skips collision but
      // re-samples the collision distance.
      if (hit_material_boundary) {
        particle_origin(pid, 0) = inter_points[pid * 3 + 0];
        particle_origin(pid, 1) = inter_points[pid * 3 + 1];
        particle_origin(pid, 2) = inter_points[pid * 3 + 2];
        particle_destination(pid, 0) = inter_points[pid * 3 + 0];
        particle_destination(pid, 1) = inter_points[pid * 3 + 1];
        particle_destination(pid, 2) = inter_points[pid * 3 + 2];
        next_elems[pid] = elem_ids[pid];
        lastExit[pid] = -2;
        material_ids[pid] = class_ids[next_elems[pid]];
      }

      ptcl_done[pid] =
          (reached_destination || hit_material_boundary) ? 1 : ptcl_done[pid];

      if (!initial) {
        if (next_elems[pid] == -1) {
          OMEGA_H_CHECK_PRINTF(
              material_ids[pid] == -1,
              "Error: next_elems[%d] is -1 but material_ids[%d] is %d\n",
              pid, pid, material_ids[pid]);
        }
      }

      if (hit_outer_boundary) {
        Omega_h::LO hit_face =
            (lastExit[pid] == -1) ? xFace[pid] : lastExit[pid];
        xFace[pid] = hit_face;
        lastExit[pid] = hit_face;
        // NOTE: Do NOT set ptcl_done here. The reflected track still needs
        // to be traced from the boundary intersection to the reflected
        // destination. Setting ptcl_done would skip that segment, losing
        // the reflected contribution from the tally.
        next_elems[pid] = elem_ids[pid];

        OMEGA_H_CHECK_PRINTF(hit_face != -1,
                             "Error: xFace[%d] is -1 but "
                             "hit_outer_boundary is true\n",
                             pid);

        auto normal = Omega_h::Vector<3>{normals[hit_face * 3 + 0],
                                         normals[hit_face * 3 + 1],
                                         normals[hit_face * 3 + 2]};
        Omega_h::Vector<3> incident_vector = {
            particle_destination(pid, 0) - inter_points[pid * 3],
            particle_destination(pid, 1) - inter_points[pid * 3 + 1],
            particle_destination(pid, 2) - inter_points[pid * 3 + 2]};

        // Specular reflection: reflected = incident - 2*(incident·normal)*normal
        // Uses remaining displacement (dest - inter_point) as incident,
        // particle direction is reflected
        // and the remaining path length is conserved.
        Omega_h::Vector<3> reflected_vector =
            incident_vector -
            2.0 * Omega_h::inner_product(incident_vector, normal) * normal;

        particle_origin(pid, 0) = inter_points[pid * 3];
        particle_origin(pid, 1) = inter_points[pid * 3 + 1];
        particle_origin(pid, 2) = inter_points[pid * 3 + 2];

        particle_destination(pid, 0) =
            particle_origin(pid, 0) + reflected_vector[0];
        particle_destination(pid, 1) =
            particle_origin(pid, 1) + reflected_vector[1];
        particle_destination(pid, 2) =
            particle_origin(pid, 2) + reflected_vector[2];
      }
    }
  };
  pumipic::parallel_for(ptcls, checkExposedEdges,
                        "apply reflective boundary condition");
}

void distributeParticlesBasesOnVolume(Omega_h::Mesh &mesh,
                                      pumitally::PPPS::kkLidView ppe,
                                      const int numPtcls) {
  OMEGA_H_CHECK_PRINTF(mesh.dim() == 3,
                       "Distributing particles based on volume is only "
                       "supported for 3D meshes, but got %dD mesh\n",
                       mesh.dim());
  o::LO ne = mesh.nelems();
  o::Real mesh_volume = volume_of_3d_mesh(mesh);
  OMEGA_H_CHECK(mesh_volume > 0.0);

  auto coords = mesh.coords();
  auto element2nodes = mesh.ask_down(o::REGION, o::VERT).ab2b;

  auto distribute_based_on_volume = OMEGA_H_LAMBDA(o::LO e) {
    auto verts = o::gather_verts<4>(element2nodes, e);
    auto vert_coords = o::gather_vectors<4, 3>(coords, verts);
    o::Real vol = volume_tet(vert_coords);
    o::Real volume_fraction = vol / mesh_volume;
    ppe[e] = std::round(numPtcls * volume_fraction);
  };
  o::parallel_for(ne, distribute_based_on_volume);

  Omega_h::LO totPtcls = 0;
  Kokkos::parallel_reduce(
      ppe.size(),
      OMEGA_H_LAMBDA(const int i, Omega_h::LO &lsum) { lsum += ppe[i]; },
      totPtcls);

  int extra_particles = numPtcls - totPtcls;
  OMEGA_H_CHECK_PRINTF(extra_particles <= mesh.nelems(),
                       "Extra particles (%d) should be less than or equal to "
                       "number of elements (%d)\n",
                       extra_particles, mesh.nelems());

  int add_remove = (extra_particles > 0) ? 1 : -1;
  auto add_or_remove_particles = OMEGA_H_LAMBDA(o::LO e) {
    ppe[e] += add_remove;
  };
  o::parallel_for(std::abs(extra_particles), add_or_remove_particles);
}

OMEGA_H_DEVICE o::Vector<3>
barycentric2real(const o::Few<o::Vector<3>, 4> &tet_verts,
                 const o::Vector<4> &bary) {
  o::Vector<3> real_coords{0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    real_coords += bary[i] * tet_verts[i];
  }
  return real_coords;
}

void initialize_uniform_source(Omega_h::Mesh &mesh,
                               Omega_h::Write<Omega_h::Real> particle_positions,
                               pumitally::PPPS::kkLidView ppe) {
  int dim = mesh.dim();
  OMEGA_H_CHECK(dim == 3);

  Omega_h::Write<Omega_h::LO> cumulative_particles(mesh.nelems() + 1, 0);
  Omega_h::LO num_particles_cumsum = 0;
  auto calculate_cumulative_number_of_particles =
      KOKKOS_LAMBDA(const int &e, Omega_h::LO &cumulative, bool is_final) {
    auto num_particles = ppe[e];
    cumulative += num_particles;
    if (is_final) {
      cumulative_particles[e + 1] = cumulative;
    }
  };
  Kokkos::parallel_scan("calculate_cumulative_number_of_particles",
                        mesh.nelems(), calculate_cumulative_number_of_particles,
                        num_particles_cumsum);

  OMEGA_H_CHECK_PRINTF(num_particles_cumsum == particle_positions.size() / 3,
                       "Total number of particles (%ld) does not match "
                       "cumulative particles (%ld)\n",
                       particle_positions.size() / 3, num_particles_cumsum);

  const auto cells2nodes = mesh.ask_down(o::REGION, o::VERT).ab2b;
  const auto coords = mesh.coords();

  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> random_pool(0);
  auto set_initial_positions = OMEGA_H_LAMBDA(const int &e) {
    auto pid_start = cumulative_particles[e];
    auto pid_end = cumulative_particles[e + 1];
    auto num_particles_in_element = pid_end - pid_start;
    OMEGA_H_CHECK(num_particles_in_element >= 0);

    for (Omega_h::LO pid = pid_start; pid < pid_end; ++pid) {
      auto gen = random_pool.get_state();
      o::Real r1 = gen.drand(0.0, 1.0);
      o::Real r2 = gen.drand(0.0, 1.0);
      o::Real r3 = gen.drand(0.0, 1.0);

      r1 = Kokkos::pow(r1, 1.0 / 3.0);
      r2 = Kokkos::sqrt(r2);
      o::Real a = 1.0 - r1;
      o::Real b = r1 * (1.0 - r2);
      o::Real c = r1 * r2 * (1.0 - r3);
      o::Real d = r1 * r2 * r3;

      o::Vector<4> random_bcc{a, b, c, d};

      random_pool.free_state(gen);

      auto verts = o::gather_verts<4>(cells2nodes, e);
      auto vert_coords = o::gather_vectors<4, 3>(coords, verts);

      auto real_loc = barycentric2real(vert_coords, random_bcc);

      particle_positions[pid * 3 + 0] = real_loc[0];
      particle_positions[pid * 3 + 1] = real_loc[1];
      particle_positions[pid * 3 + 2] = real_loc[2];
    }
  };
  o::parallel_for(mesh.nelems(), set_initial_positions);
}

void ApplyVacuumBC(const Omega_h::Mesh &mesh, PPPS *ptcls,
                   const Omega_h::Write<Omega_h::LO> &elem_ids,
                   const Omega_h::Write<Omega_h::LO> &next_elems,
                   const Omega_h::Write<Omega_h::LO> &ptcl_done,
                   const Omega_h::Write<Omega_h::LO> &last_exit,
                   const Omega_h::Write<Omega_h::LO> &x_face,
                   const Omega_h::Write<Omega_h::Real> &inter_points) {

  const auto particle_destination = ptcls->get<1>();
  auto check_exposed_edges =
      PS_LAMBDA(const int e, const int pid, const int mask) {
    if (mask > 0 && !ptcl_done[pid]) {
      const bool reached_destination = (last_exit[pid] == -1);
      const bool hit_boundary =
          ((next_elems[pid] == -1) && (elem_ids[pid] != -1));
      ptcl_done[pid] =
          (reached_destination || hit_boundary) ? 1 : ptcl_done[pid];

      if (hit_boundary) {
        x_face[pid] = last_exit[pid];
        particle_destination(pid, 0) = inter_points[pid * 3];
        particle_destination(pid, 1) = inter_points[pid * 3 + 1];
        particle_destination(pid, 2) = inter_points[pid * 3 + 2];
      }
    }
  };
  pumipic::parallel_for(ptcls, check_exposed_edges,
                        "apply vacumm boundary condition");
}

// ==========================================================================
// Helper: allocate a DynRankView with runtime rank via dispatch
// ==========================================================================
namespace {
Kokkos::DynRankView<double, PPExeSpace>
allocate_tally_view(const std::string &label, Omega_h::LO spatial_extent,
                    const std::vector<uint> &bins) {
  switch (bins.size()) {
  case 1:
    return Kokkos::DynRankView<double, PPExeSpace>(label, spatial_extent, bins[0]);
  case 2:
    return Kokkos::DynRankView<double, PPExeSpace>(label, spatial_extent, bins[0], bins[1]);
  case 3:
    return Kokkos::DynRankView<double, PPExeSpace>(label, spatial_extent, bins[0], bins[1], bins[2]);
  case 4:
    return Kokkos::DynRankView<double, PPExeSpace>(label, spatial_extent, bins[0], bins[1], bins[2], bins[3]);
  default:
    throw std::runtime_error(
        "Unsupported number of filter dimensions: " +
        std::to_string(bins.size()));
  }
}
} // namespace

// ==========================================================================
// ParticleAtElemBoundary Implementation
// ==========================================================================

ParticleAtElemBoundary::ParticleAtElemBoundary(const Omega_h::LO num_elements,
                                               const Omega_h::LO _num_vertices,
                                               const Omega_h::LO capacity)
    : is_initial_track(true), nelem(num_elements),
      prev_xpoint(capacity * 3, 0.0, "prev_xpoint"),
      alpha_(capacity, 1.0, "alpha"),
      multi_dim_tallies_active(false), active_n_filters(0),
      num_vertices(_num_vertices),
      boundary_condition(BoundaryCondition::VACUUM) {
  printf("[INFO] Particle handler at boundary with %d elements, %d vertices, "
         "and %d x points size (3 * n_particles)\n",
         nelem, num_vertices, prev_xpoint.size());
}

void ParticleAtElemBoundary::AddElementTally(
    const std::vector<uint> &number_of_non_spatial_filter_bins) {

  OMEGA_H_CHECK(!element_tally_called); // call-once guard
  OMEGA_H_CHECK(number_of_non_spatial_filter_bins.size() > 0);

  element_tally_spec =
      TallySpec(number_of_non_spatial_filter_bins);

  element_tallies = allocate_tally_view(
      "element_tallies", nelem, number_of_non_spatial_filter_bins);
  Kokkos::deep_copy(element_tallies, 0.0);

  active_n_filters = element_tally_spec.GetNumFilters();
  multi_dim_tallies_active = true;
  element_tally_called = true;

  printf("[INFO] Element tally registered: nelem=%d, n_filters=%u, "
         "total_filter_bins=%u\n",
         nelem, element_tally_spec.GetNumFilters(),
         element_tally_spec.total_filter_bins);
}

void ParticleAtElemBoundary::AddNodeTally(
    const std::vector<uint> &number_of_non_spatial_filter_bins) {

  OMEGA_H_CHECK(!node_tally_called); // call-once guard
  OMEGA_H_CHECK(number_of_non_spatial_filter_bins.size() > 0);
  OMEGA_H_CHECK(num_vertices > 0);

  node_tally_spec = TallySpec(number_of_non_spatial_filter_bins);

  node_tallies = allocate_tally_view(
      "node_tallies", num_vertices, number_of_non_spatial_filter_bins);
  Kokkos::deep_copy(node_tallies, 0.0);

  active_n_filters = node_tally_spec.GetNumFilters();
  multi_dim_tallies_active = true;
  node_tally_called = true;

  printf("[INFO] Node tally registered: nvertices=%d, n_filters=%u, "
         "total_filter_bins=%u\n",
         num_vertices, node_tally_spec.GetNumFilters(),
         node_tally_spec.total_filter_bins);
}

void ParticleAtElemBoundary::SetFilterBins(const std::vector<uint> &bins) {
  uint n_filters = active_n_filters;

  // prev_xpoint has size capacity * 3, so capacity = prev_xpoint.size() / 3
  Omega_h::LO capacity = static_cast<Omega_h::LO>(prev_xpoint.size() / 3);
  size_t required_size = static_cast<size_t>(capacity) * n_filters;

  if (static_cast<size_t>(filter_bins_dev.size()) != required_size) {
    filter_bins_dev = Omega_h::Write<Omega_h::LO>(required_size, 0,
                                                   "filter_bins_dev");
  }

  // Copy host bins to device
  Kokkos::View<const uint *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      bins_host(bins.data(), required_size);
  Kokkos::View<Omega_h::LO *, PPExeSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      bins_dev(filter_bins_dev.data(), filter_bins_dev.size());
  Kokkos::deep_copy(bins_dev, bins_host);
}

void ParticleAtElemBoundary::SetFilterBins(const Kokkos::View<uint *> &bins) {
  uint n_filters = active_n_filters;

  Omega_h::LO capacity = static_cast<Omega_h::LO>(prev_xpoint.size() / 3);
  size_t required_size = static_cast<size_t>(capacity) * n_filters;

  if (static_cast<size_t>(filter_bins_dev.size()) != required_size) {
    filter_bins_dev = Omega_h::Write<Omega_h::LO>(required_size, 0,
                                                   "filter_bins_dev");
  }

  // Direct device-to-device copy (avoids host round-trip)
  Kokkos::View<Omega_h::LO *, PPExeSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      bins_dev(filter_bins_dev.data(), bins.size());
  Kokkos::deep_copy(bins_dev, bins);
}

void ParticleAtElemBoundary::SetBoundaryCondition(BoundaryCondition bc,
                                                  Omega_h::Mesh &mesh) {
  boundary_condition = bc;
  if (bc == BoundaryCondition::REFLECTIVE) {
    if (!mesh.has_tag(Omega_h::FACE, "normals")) {
      compute_boundary_normals(mesh);
      printf("[INFO] Computed boundary normals for reflective BC\n");
    }
    // Store normals in the handler so they survive mesh copies.
    // The ParticleTracer holds a copy of the mesh, and tags added
    // after construction are not visible through that copy.
    boundary_normals = mesh.get_array<Omega_h::Real>(Omega_h::FACE, "normals");
    printf("[INFO] Boundary condition set to REFLECTIVE (specular)\n");
  } else {
    printf("[INFO] Boundary condition set to VACUUM\n");
  }
}

// ==========================================================================
// operator() — entry point called by ParticleTracer at each boundary crossing
// ==========================================================================

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

  switch (boundary_condition) {
  case BoundaryCondition::VACUUM:
    ApplyVacuumBC(mesh, ptcls, elem_ids, next_elems, ptcl_done, last_exit,
                  inter_faces, inter_points);
    break;
  case BoundaryCondition::REFLECTIVE: {
    Omega_h::Write<int> mat_ids(ptcls->capacity(), 0, "material_ids");
    apply_reflection_boundary_condition(mesh, ptcls, elem_ids, next_elems,
                                        ptcl_done, last_exit, inter_faces,
                                        inter_points, mat_ids,
                                        is_initial_track, boundary_normals);
    break;
  }
  }

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

// ==========================================================================
// EvaluateFlux — with multi-dimensional tally accumulation
// ==========================================================================

void ParticleAtElemBoundary::EvaluateFlux(
    PPPS *ptcls, const Omega_h::Write<Omega_h::Real> &xpoints,
    const Omega_h::Write<Omega_h::LO> &elem_ids,
    const Omega_h::Write<Omega_h::LO> &ptcl_done) const {
  const auto prev_xpoint_l = prev_xpoint;
  const auto in_flight = ptcls->get<3>();
  const auto p_wgt = ptcls->get<4>();
  const auto &xpoints_l = xpoints;

  // Multi-dimensional tally: capture DynRankViews (cheap handle copy)
  const bool has_elem = element_tally_spec.is_initialized;
  const bool has_node = node_tally_spec.is_initialized;
  const auto elem_view = element_tallies;
  const auto node_view = node_tallies;

  // Filter bins device array — size [capacity * n_filters]
  const auto filt_bins = filter_bins_dev;
  const uint n_filt = active_n_filters;

  // Alpha weight — per-particle multiplier for tally contribution
  const auto alpha_l = alpha_;

  // Element-to-vertex connectivity for node tallies (size: nelem*4 for tets)
  const auto e2v = elem2vert;

  auto evaluate_flux =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if ((mask > 0) && (in_flight(pid) == 1) && !ptcl_done[pid]) {
      const Omega_h::Vector<3> dest = {xpoints_l[pid * 3 + 0],
                                       xpoints_l[pid * 3 + 1],
                                       xpoints_l[pid * 3 + 2]};
      const Omega_h::Vector<3> orig = {prev_xpoint_l[pid * 3 + 0],
                                       prev_xpoint_l[pid * 3 + 1],
                                       prev_xpoint_l[pid * 3 + 2]};

      const Omega_h::Real segment_length = Omega_h::norm(dest - orig);
      const Omega_h::Real contribution = segment_length * p_wgt(pid) * alpha_l[pid];

      // Multi-dimensional tallies via DynRankView operator()
      // rank = 1 (spatial) + n_filt; switch dispatches correct number of indices
      if (has_elem) {
        Omega_h::LO eid = elem_ids[pid];
        switch (n_filt) {
        case 1:
          Kokkos::atomic_add(
              &elem_view(eid, filt_bins[pid * n_filt + 0]),
              contribution);
          break;
        case 2:
          Kokkos::atomic_add(
              &elem_view(eid, filt_bins[pid * n_filt + 0],
                         filt_bins[pid * n_filt + 1]),
              contribution);
          break;
        case 3:
          Kokkos::atomic_add(
              &elem_view(eid, filt_bins[pid * n_filt + 0],
                         filt_bins[pid * n_filt + 1],
                         filt_bins[pid * n_filt + 2]),
              contribution);
          break;
        case 4:
          Kokkos::atomic_add(
              &elem_view(eid, filt_bins[pid * n_filt + 0],
                         filt_bins[pid * n_filt + 1],
                         filt_bins[pid * n_filt + 2],
                         filt_bins[pid * n_filt + 3]),
              contribution);
          break;
        default:
          break;
        }
      }

      if (has_node) {
        // Distribute to the 4 vertices of the tet element (equal split)
        Omega_h::Real vert_contrib = contribution / 4.0;
        Omega_h::LO eid = elem_ids[pid];
        Omega_h::LO v0 = e2v[eid * 4 + 0];
        Omega_h::LO v1 = e2v[eid * 4 + 1];
        Omega_h::LO v2 = e2v[eid * 4 + 2];
        Omega_h::LO v3 = e2v[eid * 4 + 3];
        switch (n_filt) {
        case 1: {
          auto b0 = filt_bins[pid * n_filt + 0];
          Kokkos::atomic_add(&node_view(v0, b0), vert_contrib);
          Kokkos::atomic_add(&node_view(v1, b0), vert_contrib);
          Kokkos::atomic_add(&node_view(v2, b0), vert_contrib);
          Kokkos::atomic_add(&node_view(v3, b0), vert_contrib);
          break;
        }
        case 2: {
          auto b0 = filt_bins[pid * n_filt + 0];
          auto b1 = filt_bins[pid * n_filt + 1];
          Kokkos::atomic_add(&node_view(v0, b0, b1), vert_contrib);
          Kokkos::atomic_add(&node_view(v1, b0, b1), vert_contrib);
          Kokkos::atomic_add(&node_view(v2, b0, b1), vert_contrib);
          Kokkos::atomic_add(&node_view(v3, b0, b1), vert_contrib);
          break;
        }
        case 3: {
          auto b0 = filt_bins[pid * n_filt + 0];
          auto b1 = filt_bins[pid * n_filt + 1];
          auto b2 = filt_bins[pid * n_filt + 2];
          Kokkos::atomic_add(&node_view(v0, b0, b1, b2), vert_contrib);
          Kokkos::atomic_add(&node_view(v1, b0, b1, b2), vert_contrib);
          Kokkos::atomic_add(&node_view(v2, b0, b1, b2), vert_contrib);
          Kokkos::atomic_add(&node_view(v3, b0, b1, b2), vert_contrib);
          break;
        }
        case 4: {
          auto b0 = filt_bins[pid * n_filt + 0];
          auto b1 = filt_bins[pid * n_filt + 1];
          auto b2 = filt_bins[pid * n_filt + 2];
          auto b3 = filt_bins[pid * n_filt + 3];
          Kokkos::atomic_add(&node_view(v0, b0, b1, b2, b3), vert_contrib);
          Kokkos::atomic_add(&node_view(v1, b0, b1, b2, b3), vert_contrib);
          Kokkos::atomic_add(&node_view(v2, b0, b1, b2, b3), vert_contrib);
          Kokkos::atomic_add(&node_view(v3, b0, b1, b2, b3), vert_contrib);
          break;
        }
        default:
          break;
        }
      }
    }
  };

  pumipic::parallel_for(ptcls, evaluate_flux, "flux evaluation loop");
}

// ==========================================================================
// FinalizeTallies — compute volumes and write tallies to VTK
// ==========================================================================

void ParticleAtElemBoundary::FinalizeTallies(
    Omega_h::Mesh &full_mesh, const std::string &filename,
    const std::vector<Omega_h::LO> &source_dist) const {
  // 1. Compute and attach element volumes
  {
    const auto &el2n = full_mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;
    const auto &coords = full_mesh.coords();
    Omega_h::Write<Omega_h::Real> tet_volumes(nelem, -1.0, "tet_volumes");

    auto compute_volume = OMEGA_H_LAMBDA(const Omega_h::LO elem_id) {
      const auto elem_verts = Omega_h::gather_verts<4>(el2n, elem_id);
      const auto elem_vert_coords =
          Omega_h::gather_vectors<4, 3>(coords, elem_verts);
      const auto b = Omega_h::simplex_basis<3, 3>(elem_vert_coords);
      tet_volumes[elem_id] = Omega_h::simplex_size_from_basis(b);
    };
    Omega_h::parallel_for(nelem, compute_volume, "compute element volumes");
    full_mesh.add_tag(Omega_h::REGION, "volume", 1, Omega_h::Reals(tet_volumes));
  }

  // 1b. Attach source distribution (particles per element) if available
  if (!source_dist.empty()) {
    OMEGA_H_CHECK_PRINTF(
        source_dist.size() == static_cast<size_t>(nelem),
        "Source distribution size (%zu) must match number of elements (%d)\n",
        source_dist.size(), nelem);
    // Convert from LO to Real so VTK can display it
    Omega_h::HostWrite<Omega_h::Real> source_host(nelem, "source_host");
    for (Omega_h::LO e = 0; e < nelem; ++e) {
      source_host[e] = static_cast<Omega_h::Real>(source_dist[e]);
    }
    Omega_h::Write<Omega_h::Real> source_write(source_host);
    full_mesh.add_tag(Omega_h::REGION, "source", 1,
                      Omega_h::Reals(source_write));
    printf("[INFO] Added source tag on REGION (particles per element)\n");
  }

  // 2. Multi-dimensional element tally: single tag with ncomps = total_filter_bins
  if (element_tally_spec.is_initialized) {
    const auto &spec = element_tally_spec;
    auto elem_data = element_tallies.data();
    Omega_h::HostWrite<Omega_h::Real> elem_host(nelem * spec.total_filter_bins,
                                                 "element_tally_host");
    for (Omega_h::LO e = 0; e < nelem; ++e) {
      for (unsigned int c = 0; c < spec.total_filter_bins; ++c) {
        elem_host[e * spec.total_filter_bins + c] =
            elem_data[e * spec.total_filter_bins + c];
      }
    }
    Omega_h::Write<Omega_h::Real> elem_write(elem_host);
    full_mesh.add_tag(Omega_h::REGION, "element_tally",
                      static_cast<int>(spec.total_filter_bins),
                      Omega_h::Reals(elem_write));
    printf("[INFO] Added element_tally tag on REGION: ncomps=%u (nelem=%d)\n",
           spec.total_filter_bins, nelem);
  }

  // 3. Multi-dimensional node tally: single tag with ncomps = total_filter_bins
  if (node_tally_spec.is_initialized) {
    const auto &spec = node_tally_spec;
    auto node_data = node_tallies.data();
    Omega_h::HostWrite<Omega_h::Real> node_host(
        num_vertices * spec.total_filter_bins, "node_tally_host");
    for (Omega_h::LO v = 0; v < num_vertices; ++v) {
      for (unsigned int c = 0; c < spec.total_filter_bins; ++c) {
        node_host[v * spec.total_filter_bins + c] =
            node_data[v * spec.total_filter_bins + c];
      }
    }
    Omega_h::Write<Omega_h::Real> node_write(node_host);
    full_mesh.add_tag(Omega_h::VERT, "node_tally",
                      static_cast<int>(spec.total_filter_bins),
                      Omega_h::Reals(node_write));
    printf("[INFO] Added node_tally tag on VERT: ncomps=%u (nvertices=%d)\n",
           spec.total_filter_bins, num_vertices);
  }

  // 4. Write VTK
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

bool PumiTallyImpl::SearchAndRebuild(const bool initial,
                                     const bool migrate) const {
  assert((is_pumipic_initialized == false && initial == true) ||
         (is_pumipic_initialized == true && initial == false));
  p_pumi_particle_at_elem_boundary_handler->MarkAsInitial(initial);
  auto orig = pumipic_ptcls->get<0>();
  auto dest = pumipic_ptcls->get<1>();
  auto pid = pumipic_ptcls->get<2>();

  if (p_picparts->mesh() == nullptr || p_picparts->mesh()->nelems() == 0) {
    fprintf(stderr, "ERROR: Mesh is empty\n");
  }

  if (!initial) {
    p_pumi_particle_at_elem_boundary_handler->UpdatePreviousXPoints(
        pumipic_ptcls.get());
  }

  const bool found_all = p_particle_tracer->search(migrate);
  if (!found_all) {
    printf(
        "ERROR: Not all particles are found. May need more loops in search\n");
  }

  // Soft-copy: point the handler's last_exit_ to the ParticleTracer's
  // last_exits_ array.  Callers (degas2) read this on the next iteration
  // to decide: -1 = reached destination, else = hit boundary face.
  p_pumi_particle_at_elem_boundary_handler->last_exit_ =
      p_particle_tracer->GetLastExits();

  return found_all;
}

std::unique_ptr<PPPS> CreateParticleDS(
    const Omega_h::Mesh &mesh, pumipic::lid_t num_ptcls,
    std::vector<Omega_h::LO> *out_ptcls_per_elem) {
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

  // Save source distribution if requested
  if (out_ptcls_per_elem) {
    out_ptcls_per_elem->resize(ne);
    auto ptcls_host = Kokkos::create_mirror_view(ptcls_per_elem);
    Kokkos::deep_copy(ptcls_host, ptcls_per_elem);
    for (Omega_h::LO i = 0; i < ne; ++i) {
      (*out_ptcls_per_elem)[i] = ptcls_host(i);
    }
  }

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

std::unique_ptr<PPPS> CreateParticleDSForRegion(
    Omega_h::Mesh &mesh, pumipic::lid_t num_ptcls, int region_id,
    std::vector<Omega_h::LO> *out_ptcls_per_elem) {
  Omega_h::Int ne = mesh.nelems();
  pumitally::PPPS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  pumitally::PPPS::kkGidView element_gids("element_gids", ne);

  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy;

  Omega_h::parallel_for(
      ne, OMEGA_H_LAMBDA(const Omega_h::LO &i) { element_gids(i) = i; });

  // Read class_id tag to identify elements in the region
  const auto class_ids = mesh.get_array<int>(Omega_h::REGION, "class_id");
  const auto coords = mesh.coords();
  const auto e2v = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

  // Compute total volume of the region (using parallel_reduce like
  // volume_of_3d_mesh does, but filtering by class_id)
  Omega_h::Real total_vol = 0.0;
  Kokkos::parallel_reduce(
      "compute region total volume", ne,
      KOKKOS_LAMBDA(const int e, Omega_h::Real &local_vol) {
        if (class_ids[e] == region_id) {
          const auto nodes = Omega_h::gather_verts<4>(e2v, e);
          const Omega_h::Few<Omega_h::Vector<3>, 4> elem_coords =
              Omega_h::gather_vectors<4, 3>(coords, nodes);
          local_vol += volume_tet(elem_coords);
        }
      },
      Kokkos::Sum<Omega_h::Real>(total_vol));

  OMEGA_H_CHECK_PRINTF(total_vol > 0.0,
                       "Region %d has zero total volume. Check that elements "
                       "with class_id=%d exist in the mesh.\n",
                       region_id, region_id);

  // Distribute particles proportionally to element volume
  Omega_h::parallel_for(
      "distribute particles in region", ne, OMEGA_H_LAMBDA(const Omega_h::LO e) {
        if (class_ids[e] == region_id) {
          const auto nodes = Omega_h::gather_verts<4>(e2v, e);
          const Omega_h::Few<Omega_h::Vector<3>, 4> elem_coords =
              Omega_h::gather_vectors<4, 3>(coords, nodes);
          Omega_h::Real vol = volume_tet(elem_coords);
          Omega_h::Real volume_fraction = vol / total_vol;
          ptcls_per_elem[e] =
              static_cast<Omega_h::LO>(std::round(num_ptcls * volume_fraction));
        } else {
          ptcls_per_elem[e] = 0;
        }
      });

  // Adjust for rounding errors: add/remove one particle at a time
  // from region elements that can absorb the change without going negative.
  Omega_h::LO tot_assigned = 0;
  Kokkos::parallel_reduce(
      ne,
      KOKKOS_LAMBDA(const int i, Omega_h::LO &lsum) {
        lsum += ptcls_per_elem[i];
      },
      tot_assigned);

  int extra = num_ptcls - tot_assigned;
  int add_remove = (extra > 0) ? 1 : -1;
  int remaining = (extra > 0) ? extra : -extra;

  // Build a host list of region element indices (cheap: region is typically
  // a small fraction of the total mesh)
  auto class_ids_host = Omega_h::HostRead<int>(class_ids);
  std::vector<Omega_h::LO> region_elems;
  region_elems.reserve(ne);
  for (Omega_h::LO e = 0; e < ne && remaining > 0; e++) {
    if (class_ids_host[e] == region_id) {
      region_elems.push_back(e);
    }
  }

  printf("[INFO] Region %d: %zu elements, total volume %.6e, assigned %d / %d "
         "particles (adjusting by %d)\n",
         region_id, region_elems.size(), total_vol, tot_assigned,
         static_cast<int>(num_ptcls), extra);

  // Distribute extra across elements round-robin, never going below 0
  if (!region_elems.empty()) {
    for (int i = 0; i < remaining; i++) {
      Omega_h::LO e = region_elems[i % region_elems.size()];
      if (add_remove > 0 || ptcls_per_elem[e] > 0) {
        Kokkos::atomic_add(&ptcls_per_elem[e], add_remove);
      }
    }
  } else if (extra > 0) {
    // No elements in region yet we have particles to assign — fatal
    OMEGA_H_CHECK_PRINTF(false,
                         "Region %d has no elements but %d particles were "
                         "requested. Cannot distribute.\n",
                         region_id, static_cast<int>(num_ptcls));
  }

  // Save source distribution if requested
  if (out_ptcls_per_elem) {
    out_ptcls_per_elem->resize(ne);
    auto ptcls_host = Kokkos::create_mirror_view(ptcls_per_elem);
    Kokkos::deep_copy(ptcls_host, ptcls_per_elem);
    for (Omega_h::LO i = 0; i < ne; ++i) {
      (*out_ptcls_per_elem)[i] = ptcls_host(i);
    }
  }

  printf("PumiPIC Using CPU for simulation...\n");
  policy =
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());

  auto ptcls = std::make_unique<pumipic::DPS<pumitally::PPParticle>>(
      policy, ne, num_ptcls, ptcls_per_elem, element_gids);

  return ptcls;
}

void InitializeParticlesInElement0(Omega_h::Mesh &mesh,
                                   pumitally::PPPS *ptcls) {
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

  auto init_loc = ptcls->get<0>();
  auto pids = ptcls->get<2>();
  auto in_fly = ptcls->get<3>();
  auto p_wgt = ptcls->get<4>();

  auto set_initial_positions =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0) {
      pids(pid) = pid;
      in_fly(pid) = 1;
      init_loc(pid, 0) = centroid_of_el0[0];
      init_loc(pid, 1) = centroid_of_el0[1];
      init_loc(pid, 2) = centroid_of_el0[2];
      p_wgt(pid) = 1.0;
    }
  };
  pumipic::parallel_for(ptcls, set_initial_positions,
                        "set is_initial_track particle positions");
}

void InitializeParticlesInRegion(Omega_h::Mesh &mesh,
                                  pumitally::PPPS *ptcls,
                                  int /*region_id*/) {
  const auto &coords = mesh.coords();
  const auto &tet2node = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

  auto init_loc = ptcls->get<0>();
  auto pids = ptcls->get<2>();
  auto in_fly = ptcls->get<3>();
  auto p_wgt = ptcls->get<4>();

  auto set_initial_positions =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0) {
      // Compute centroid of the current element
      const auto nodes = Omega_h::gather_verts<4>(tet2node, e);
      const Omega_h::Few<Omega_h::Vector<3>, 4> tet_node_coords =
          Omega_h::gather_vectors<4, 3>(coords, nodes);
      Omega_h::Vector<3> centroid = o::average(tet_node_coords);

      pids(pid) = pid;
      in_fly(pid) = 1;
      init_loc(pid, 0) = centroid[0];
      init_loc(pid, 1) = centroid[1];
      init_loc(pid, 2) = centroid[2];
      p_wgt(pid) = 1.0;
    }
  };
  pumipic::parallel_for(ptcls, set_initial_positions,
                        "set region particle positions");
}

Omega_h::Mesh PumiTallyImpl::PartitionMesh() {
  const Omega_h::Write<Omega_h::LO> owners(full_mesh.nelems(), 0, "owners");
  p_picparts = std::make_unique<pumipic::Mesh>(full_mesh, Omega_h::LOs(owners));
  printf("PumiPIC mesh partitioned\n");

  return *p_picparts->mesh();
}

void PumiTallyImpl::InitializePUMIParticleStructure(Omega_h::Mesh &mesh) {
  pumipic_ptcls = CreateParticleDS(mesh, num_particles, &source_ptcls_per_elem);
  InitializeParticlesInElement0(mesh, pumipic_ptcls.get());
  p_pumi_particle_at_elem_boundary_handler =
      std::make_unique<pumitally::ParticleAtElemBoundary>(
          mesh.nelems(), mesh.nverts(), pumipic_ptcls->capacity());

  // Cache element-to-vertex connectivity for node tally support
  p_pumi_particle_at_elem_boundary_handler->elem2vert =
      mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

  printf("PumiPIC Mesh and data structure created with %d and %d as particle "
         "structure capacity\n",
         p_picparts->mesh()->nelems(), pumipic_ptcls->capacity());
}

void PumiTallyImpl::InitializePUMIParticleStructureForRegion(
    Omega_h::Mesh &mesh, int region_id) {
  pumipic_ptcls =
      CreateParticleDSForRegion(mesh, num_particles, region_id,
                                &source_ptcls_per_elem);
  InitializeParticlesInRegion(mesh, pumipic_ptcls.get(), region_id);
  p_pumi_particle_at_elem_boundary_handler =
      std::make_unique<pumitally::ParticleAtElemBoundary>(
          mesh.nelems(), mesh.nverts(), pumipic_ptcls->capacity());

  // Cache element-to-vertex connectivity for node tally support
  p_pumi_particle_at_elem_boundary_handler->elem2vert =
      mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

  printf("PumiPIC Mesh and data structure created with %d and %d as particle "
         "structure capacity (region %d)\n",
         p_picparts->mesh()->nelems(), pumipic_ptcls->capacity(), region_id);
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
