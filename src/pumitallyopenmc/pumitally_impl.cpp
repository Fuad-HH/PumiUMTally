//
// Created by Fuad Hasan on 12/3/24.
//

#include "pumitally_impl.tpp"
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>
#include <chrono>

namespace pumiinopenmc {

void distributeParticlesBasesOnVolume(Omega_h::Mesh &mesh,
                                      pumiinopenmc::PPPS::kkLidView ppe,
                                      const int numPtcls);
void initialize_uniform_source(Omega_h::Mesh &mesh,
                               Omega_h::Write<Omega_h::Real> particle_positions,
                               pumiinopenmc::PPPS::kkLidView ppe);

void TallyTimes::print_times() const {
  printf("\n");
  printf("[TIME] Initialization time     : %f seconds\n", initialization_time);
  printf("[TIME] Total time to tally     : %f seconds\n", total_time_to_tally);
  printf("[TIME] VTK file write time     : %f seconds\n", vtk_file_write_time);
  printf("[TIME] Total PumiPic time      : %f seconds\n",
         initialization_time + total_time_to_tally + vtk_file_write_time);
}

PumiTallyImpl::PumiTallyImpl(std::string &mesh_filename, int64_t num_particles,
                             int &argc, char **&argv,
                             SourceDistribution source_dist) {
  pumi_ps_size = num_particles;
  oh_mesh_fname = mesh_filename;

  device_pos_buffer_ =
      Omega_h::Write<Omega_h::Real>(pumi_ps_size * 3, 0.0, "device_pos_buffer");
  device_in_adv_que_ =
      Omega_h::Write<Omega_h::I8>(pumi_ps_size, 0, "device_in_adv_que");
  weights_ = Omega_h::Write<Omega_h::Real>(pumi_ps_size, 0.0, "weights");
  groups_ = Omega_h::Write<int>(pumi_ps_size, 0, "groups");

  // todo can track lengths be here?

  load_pumipic_mesh_and_init_particles(argc, argv, source_dist);
  if (source_dist == SourceDistribution::ZERO) {
    start_pumi_particles_in_0th_element(*p_picparts_->mesh(),
                                        pumipic_ptcls.get());
  }

  p_particle_tracer_ = std::make_unique<
      ParticleTracer<PPParticle, pumiinopenmc::PumiParticleAtElemBoundary>>(
      *p_picparts_, pumipic_ptcls.get(),
      *p_pumi_particle_at_elem_boundary_handler, 1e-8);
}

void PumiTallyImpl::initialize_particle_location(
    double *init_particle_positions, int64_t size) {
  // copy to host buffer
  assert(size == pumi_ps_size * 3);
  copy_data_to_device(init_particle_positions);
  search_initial_elements();
  // TODO Get total initial particle weight
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::move_to_next_location(double *particle_destinations,
                                          int8_t *flying, double *weights,
                                          int *groups, int *material_ids,
                                          int64_t size) {
  assert(size == pumi_ps_size * 3);

  // copy to device buffer
  copy_data_to_device(particle_destinations);
  // copy fly to device buffer
  copy_and_reset_flying_flag(flying);
  copy_weights(weights);
  copy_groups(groups);

  // copy position buffer ps
  auto particle_dest = pumipic_ptcls->get<1>();
  auto in_flight = pumipic_ptcls->get<3>();

  int64_t pumi_ps_size_ = pumi_ps_size;
  const auto &device_pos_buffer_l = device_pos_buffer_;
  const auto &device_in_adv_que_l = device_in_adv_que_;

  auto set_particle_dest =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && pid < pumi_ps_size_) {
      particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
      particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
      particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];

      // everyone is in flight for this initial search
      in_flight(pid) = device_in_adv_que_l[pid];
    }
  };
  pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest,
                        "set particle position as dest");

  bool migrate = iter_count_ % 100 == 0;
  iter_count_++;
  search_and_rebuild(false, migrate);
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
  copy_last_location(particle_destinations, size);
  copy_material_ids(material_ids, size / 3);
}

void PumiTallyImpl::copy_last_location(double *particle_destination,
                                       int64_t size) {
  auto last_location = p_pumi_particle_at_elem_boundary_handler->prev_xpoint_;
  OMEGA_H_CHECK_PRINTF(last_location.size() >= size,
                       "last_location size %d is not greater to %ld\n",
                       last_location.size(), size);
  Kokkos::View<Omega_h::Real *, PPExeSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      device_last_location_view(last_location.data(), size);
  Kokkos::View<Omega_h::Real *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_last_location_view(particle_destination, size);

  Kokkos::deep_copy(host_last_location_view, device_last_location_view);
}

void PumiTallyImpl::copy_material_ids(int *material_ids, int64_t size) {
  auto material_ids_l = p_pumi_particle_at_elem_boundary_handler->material_ids_;
  OMEGA_H_CHECK_PRINTF(material_ids_l.size() >= size,
                       "material_ids_l size %d is not greater to %ld\n",
                       material_ids_l.size(), size);
  Kokkos::View<int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_material_ids_view(material_ids, size);
  Kokkos::View<int *, PPExeSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      device_material_ids_view(material_ids_l.data(), size);

  Kokkos::deep_copy(host_material_ids_view, device_material_ids_view);
}

void PumiTallyImpl::write_pumi_tally_mesh() {
  p_pumi_particle_at_elem_boundary_handler->finalizeAndWritePumiFlux(
      full_mesh_, "fluxresult.vtk");
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::copy_and_reset_flying_flag(int8_t *flying) {
  // todo get the size too
  auto device_in_adv_que_l = device_in_adv_que_;
  Kokkos::View<Omega_h::I8 *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_flying_view(flying, pumi_ps_size);
  Kokkos::View<Omega_h::I8 *, PPExeSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      device_flying_view(device_in_adv_que_l.data(),
                         device_in_adv_que_l.size());
  Kokkos::deep_copy(device_flying_view, host_flying_view);

  for (int64_t pid = 0; pid < pumi_ps_size; ++pid) {
    // reset flying flag to zero
    flying[pid] = 0;
  }
}

void PumiTallyImpl::copy_weights(double *weights) {
  auto weights_l = weights_;
  Kokkos::View<Omega_h::Real *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_weights_view(weights, pumi_ps_size);
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

void PumiTallyImpl::copy_groups(int *groups) {
  auto groups_l = groups_;
  Kokkos::View<int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_groups_view(groups, pumi_ps_size);
  Kokkos::View<int *, PPExeSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      device_groups_view(groups_l.data(), groups_l.size());

  Kokkos::deep_copy(device_groups_view, host_groups_view);
  auto p_groups = pumipic_ptcls->get<5>();
  auto copy_particle_groups =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    p_groups(pid) = groups_l[pid];
  };
  pumipic::parallel_for(pumipic_ptcls.get(), copy_particle_groups,
                        "copy particle groups");
}

void PumiTallyImpl::search_initial_elements() { // assign the location to ptcl
                                                // dest
  auto particle_dest = pumipic_ptcls->get<1>();
  auto in_flight = pumipic_ptcls->get<3>();

  int64_t pumi_ps_size_l = pumi_ps_size;
  const auto &device_pos_buffer_l = device_pos_buffer_;

  auto set_particle_dest =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && pid < pumi_ps_size_l) {
      particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
      particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
      particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];

      // everyone is in flight for this initial search
      in_flight(pid) = 1;
    }
  };
  pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest,
                        "set initial position as dest");

  // *initial* build and search to find the initial elements of the particles
  search_and_rebuild(true, true);
  is_pumipic_initialized = true;
}

void PumiTallyImpl::copy_data_to_device(double *init_particle_positions) {
  // fixme it should get size too to avoid memory error
  Kokkos::View<Omega_h::Real *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_pos_view(init_particle_positions, pumi_ps_size * 3);

  Kokkos::View<Omega_h::Real *, PPExeSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      device_pos_view(device_pos_buffer_.data(), pumi_ps_size * 3);

  Kokkos::deep_copy(device_pos_view, host_pos_view);
}

// methods for PumiTallyImpl and PumiParticleAtElemBoundary

void PumiTallyImpl::init_pumi_libs(int &argc, char **&argv) {
  pp_lib = std::make_unique<pumipic::Library>(&argc, &argv);
  oh_lib = pp_lib->omega_h_lib();
}

[[deprecated("Use move_to_next_element which is appropriate for new search "
             "class in pumipic. [!Note] It my show this even though it is not "
             "used due to template instantiation.")]]
void pp_move_to_new_element(Omega_h::Mesh &mesh, PPPS *ptcls,
                            Omega_h::Write<Omega_h::LO> &elem_ids,
                            Omega_h::Write<Omega_h::LO> &ptcl_done,
                            Omega_h::Write<Omega_h::LO> &lastExit) {
  const int dim = mesh.dim();
  const auto &face2elems = mesh.ask_up(dim - 1, dim);
  const auto &face2elemElem = face2elems.ab2b;
  const auto &face2elemOffset = face2elems.a2ab;
  const auto in_flight = ptcls->get<3>();

  auto set_next_element =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && !ptcl_done[pid] && in_flight(pid)) {
      auto searchElm = elem_ids[pid];
      auto bridge = lastExit[pid];
      auto e2f_first = face2elemOffset[bridge];
      auto e2f_last = face2elemOffset[bridge + 1];
      auto upFaces = e2f_last - e2f_first;
      assert(upFaces == 2);
      auto faceA = face2elemElem[e2f_first];
      auto faceB = face2elemElem[e2f_first + 1];
      assert(faceA != faceB);
      assert(faceA == searchElm || faceB == searchElm);
      auto nextElm = (faceA == searchElm) ? faceB : faceA;
      elem_ids[pid] = nextElm;
    }
  };
  parallel_for(ptcls, set_next_element, "pumipic_set_next_element");
}

void move_to_next_element(PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
                          Omega_h::Write<Omega_h::LO> &next_elems) {
  auto in_flight = ptcls->get<3>();
  auto move_to_next = PS_LAMBDA(const int e, const int pid, const int mask) {
    // move only if particle in flight and not leaving the domain
    if (mask > 0 && in_flight(pid) && next_elems[pid] != -1) {
      elem_ids[pid] = next_elems[pid];
    }
  };
  pumipic::parallel_for(ptcls, move_to_next, "move to next element");
}

void apply_boundary_condition(Omega_h::Mesh &mesh, PPPS *ptcls,
                              Omega_h::Write<Omega_h::LO> &elem_ids,
                              Omega_h::Write<Omega_h::LO> &next_elems,
                              Omega_h::Write<Omega_h::LO> &ptcl_done,
                              Omega_h::Write<Omega_h::LO> &lastExit,
                              Omega_h::Write<Omega_h::LO> &xFace,
                              Omega_h::Write<Omega_h::Real> &inter_points,
                              Omega_h::Write<int> material_ids, bool initial) {

  // TODO: make this a member variable of the struct
  auto particle_destination = ptcls->get<1>();
  const auto class_ids = mesh.get_array<int>(3, "class_id");
  auto checkExposedEdges =
      PS_LAMBDA(const int e, const int pid, const int mask) {
    if (mask > 0 && !ptcl_done[pid]) {
      bool reached_destination = (lastExit[pid] == -1);
      bool hit_boundary = ((next_elems[pid] == -1) && (elem_ids[pid] != -1));

      // for the initial run, we need to find the initial position of the
      // particles
      if (!initial) { // stop at geometry boundary
        if (next_elems[pid] != -1) {
          if (class_ids[elem_ids[pid]] !=
              class_ids[next_elems[pid]]) { // particle crosses geometry
                                            // boundary
            hit_boundary = true;
            material_ids[pid] = class_ids[next_elems[pid]];
          }
        } else {
          material_ids[pid] = -1; // no material id if not in an element
        }
      }

      ptcl_done[pid] =
          (reached_destination || hit_boundary) ? 1 : ptcl_done[pid];
      // assert that if the next element is -1, then the material id is -1
      if (!initial) {
        if (next_elems[pid] == -1) {
          OMEGA_H_CHECK_PRINTF(
              material_ids[pid] == -1,
              "Error: next_elems[%d] is -1 but material_ids[%d] "
              "is %d\n",
              pid, pid, material_ids[pid]);
        }
        // printf("Pid %d next element %d, elem id %d, material id %d\n",
        //        pid, next_elems[pid], elem_ids[pid], material_ids[pid]);
      }

      if (hit_boundary) { // just reached the boundary
        // printf("Moving particle %4d (from -> to): element (%5d -> %5d)
        // Material (%3d -> %3d)\n",
        //      pid, elem_ids[pid], next_elems[pid],
        //      class_ids[elem_ids[pid]], material_ids[pid]);
        xFace[pid] = lastExit[pid];
        // particle reaches the boundary
        particle_destination(pid, 0) = inter_points[pid * 3];
        particle_destination(pid, 1) = inter_points[pid * 3 + 1];
        particle_destination(pid, 2) = inter_points[pid * 3 + 2];
      }
    }
  };
  pumipic::parallel_for(ptcls, checkExposedEdges,
                        "apply vacumm boundary condition");
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

  // calculate the normals for the exposed edges
  auto calculate_normals = OMEGA_H_LAMBDA(const Omega_h::LO &face_id) {
    if (exposed_edges[face_id]) {
      // get this face's nodes
      const auto face_nodes = Omega_h::gather_verts<3>(face2nodes, face_id);
      const auto face_coords =
          Omega_h::gather_vectors<3, 3>(coords, face_nodes);
      const auto normal = Omega_h::cross(
          face_coords[1] - face_coords[0],
          face_coords[2] - face_coords[0]); // cross product to get normal

      const auto norm = Omega_h::norm(normal);

      // fix direction of the normal: get the fourth node of the element and
      // check with it
      const auto elem_id =
          face2elems[face2elemsOffset[face_id]]; // edge sides only have one
                                                 // element
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

      // check if the normal points towards the fourth node
      Omega_h::Vector<3> fourth_2_face_vector = {
          fourth_node_coord[0] - face_coords[0][0],
          fourth_node_coord[1] - face_coords[0][1],
          fourth_node_coord[2] - face_coords[0][2]};

      Omega_h::Vector<3> inner_norm;
      if (Omega_h::inner_product(normal, fourth_2_face_vector) < 0) {
        // flip the normal if it points away from the fourth node
        for (int i = 0; i < 3; ++i) {
          inner_norm[i] = -normal[i] / norm;
        }
      } else {
        for (int i = 0; i < 3; ++i) {
          inner_norm[i] = normal[i] / norm;
        }
      }
      // store the normal in the normals array
      normals[face_id * 3 + 0] = inner_norm[0];
      normals[face_id * 3 + 1] = inner_norm[1];
      normals[face_id * 3 + 2] = inner_norm[2];
    }
  };
  Omega_h::parallel_for(mesh.nfaces(), calculate_normals,
                        "compute boundary normals");
  mesh.add_tag(Omega_h::FACE, "normals", 3, Omega_h::Reals(normals));
}

void apply_reflection_boundary_condition(
    Omega_h::Mesh &mesh, PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
    Omega_h::Write<Omega_h::LO> &next_elems,
    Omega_h::Write<Omega_h::LO> &ptcl_done,
    Omega_h::Write<Omega_h::LO> &lastExit, Omega_h::Write<Omega_h::LO> &xFace,
    Omega_h::Write<Omega_h::Real> &inter_points,
    Omega_h::Write<int> material_ids, bool initial) {

  // TODO: make this a member variable of the struct
  auto particle_destination = ptcls->get<1>();
  auto particle_origin = ptcls->get<0>();

  const auto class_ids = mesh.get_array<int>(3, "class_id");
  const auto normals = mesh.get_array<Omega_h::Real>(Omega_h::FACE, "normals");

  auto checkExposedEdges =
      PS_LAMBDA(const int e, const int pid, const int mask) {
    if (mask > 0 && !ptcl_done[pid]) {
      bool reached_destination = (lastExit[pid] == -1);
      bool hit_outer_boundary =
          ((next_elems[pid] == -1) && (elem_ids[pid] != -1));

      // for the initial run, we need to find the initial position of the
      // particles
      bool hit_material_boundary = false;
      if (!initial) { // stop at geometry boundary
        if (next_elems[pid] != -1) {
          if (class_ids[elem_ids[pid]] !=
              class_ids[next_elems[pid]]) { // particle crosses geometry
            // boundary
            hit_material_boundary = true;
            material_ids[pid] = class_ids[next_elems[pid]];
          }
        } else {
          material_ids[pid] = -1; // no material id if not in an element
        }
      }

      ptcl_done[pid] =
          (reached_destination || hit_material_boundary) ? 1 : ptcl_done[pid];
      // assert that if the next element is -1, then the material id is -1
      if (!initial) {
        if (next_elems[pid] == -1) {
          OMEGA_H_CHECK_PRINTF(
              material_ids[pid] == -1,
              "Error: next_elems[%d] is -1 but material_ids[%d] "
              "is %d\n",
              pid, pid, material_ids[pid]);
        }
        // printf("Pid %d next element %d, elem id %d, material id %d\n",
        //        pid, next_elems[pid], elem_ids[pid], material_ids[pid]);
      }

      // reflective boundary condition
      if (hit_outer_boundary) { // just reached the boundary
        // printf("Moving particle %4d (from -> to): element (%5d -> %5d)
        // Material (%3d -> %3d)\n",
        //      pid, elem_ids[pid], next_elems[pid],
        //      class_ids[elem_ids[pid]], material_ids[pid]);
        xFace[pid] = lastExit[pid];
        OMEGA_H_CHECK_PRINTF(lastExit[pid] != -1,
                             "Error: lastExit[%d] is -1 but "
                             "hit_outer_boundary is true\n",
                             pid);

        // change direction
        auto normal = Omega_h::Vector<3>{normals[lastExit[pid] * 3 + 0],
                                         normals[lastExit[pid] * 3 + 1],
                                         normals[lastExit[pid] * 3 + 2]};
        Omega_h::Vector<3> particle_direction = {
            particle_destination(pid, 0) - particle_origin(pid, 0),
            particle_destination(pid, 1) - particle_origin(pid, 1),
            particle_destination(pid, 2) - particle_origin(pid, 2)};
        // reflect the particle direction
        Omega_h::Vector<3> reflected_direction =
            particle_direction -
            2.0 * Omega_h::inner_product(particle_direction, normal) * normal;

        // change the particle's position
        // particle reaches the boundary
        particle_origin(pid, 0) = inter_points[pid * 3];
        particle_origin(pid, 1) = inter_points[pid * 3 + 1];
        particle_origin(pid, 2) = inter_points[pid * 3 + 2];

        particle_destination(pid, 0) =
            particle_origin(pid, 0) + reflected_direction[0];
        particle_destination(pid, 1) =
            particle_origin(pid, 1) + reflected_direction[1];
        particle_destination(pid, 2) =
            particle_origin(pid, 2) + reflected_direction[2];
      }
    }
  };
  pumipic::parallel_for(ptcls, checkExposedEdges,
                        "apply reflective boundary condition");
}

void PumiParticleAtElemBoundary::initialize_flux_array(size_t nelems,
                                                       size_t nEgroups) {
  Kokkos::resize(flux_, nelems, nEgroups, 3);
  auto flux_l = flux_;
  OMEGA_H_CHECK(flux_l.extent(0) == nelems);
  OMEGA_H_CHECK(flux_l.extent(1) == nEgroups);
  OMEGA_H_CHECK(flux_l.extent(2) == 3);
}

PumiParticleAtElemBoundary::PumiParticleAtElemBoundary(size_t nelems,
                                                       size_t capacity)
    : prev_xpoint_(capacity * 3, 0.0, "prev_xpoint"),
      material_ids_(capacity, -1, "material_ids"),
      total_tracklength_(capacity, 0.0, "total_tracklength"), initial_(true) {
  initialize_flux_array(nelems, 2); // FIXME: hardcoded 2 groups
  printf("[INFO] Particle handler at boundary with %d elements and %d "
         "x points size (3 * n_particles)\n",
         flux_.size(), prev_xpoint_.size());
}

void PumiParticleAtElemBoundary::operator()(
    Omega_h::Mesh &mesh, pumiinopenmc::PPPS *ptcls,
    Omega_h::Write<Omega_h::LO> &elem_ids,
    Omega_h::Write<Omega_h::LO> &next_elems,
    Omega_h::Write<Omega_h::LO> &inter_faces,
    Omega_h::Write<Omega_h::LO> &lastExit,
    Omega_h::Write<Omega_h::Real> &inter_points,
    Omega_h::Write<Omega_h::LO> &ptcl_done,
    typeof(ptcls->get<0>()) origin_segment,
    typeof(ptcls->get<1>()) dest_segment) {
  if (!initial_) {
    evaluateFlux(ptcls, inter_points, elem_ids, ptcl_done);
    updatePrevXPoint(inter_points);
    updateLastExit(lastExit);
  }
  apply_reflection_boundary_condition(mesh, ptcls, elem_ids, next_elems,
                                      ptcl_done, lastExit, inter_faces,
                                      inter_points, material_ids_, initial_);
  move_to_next_element(ptcls, elem_ids, next_elems);
}

void PumiParticleAtElemBoundary::mark_initial_as(bool initial) {
  initial_ = initial;
}

void PumiParticleAtElemBoundary::updatePrevXPoint(
    Omega_h::Write<Omega_h::Real> &xpoints) {
  OMEGA_H_CHECK_PRINTF(xpoints.size() <= prev_xpoint_.size() &&
                           prev_xpoint_.size() != 0,
                       "xpoints size %d is greater than prev_xpoint size %d\n",
                       xpoints.size(), prev_xpoint_.size());
  auto &prev_xpoint = prev_xpoint_;
  auto update = OMEGA_H_LAMBDA(Omega_h::LO i) { prev_xpoint[i] = xpoints[i]; };
  Omega_h::parallel_for(xpoints.size(), update, "update previous xpoints");
}

void PumiParticleAtElemBoundary::updatePrevXPoint(PPPS *ptcls) {
  // todo add checks of size
  auto prev_xpoints_l = prev_xpoint_;
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

void PumiParticleAtElemBoundary::evaluateFlux(
    PPPS *ptcls, Omega_h::Write<Omega_h::Real> xpoints,
    Omega_h::Write<Omega_h::LO> elem_ids,
    Omega_h::Write<Omega_h::LO> ptcl_done) {
  // Omega_h::Real total_particles = ptcls->nPtcls();
  auto prev_xpoint = prev_xpoint_;
  auto flux = flux_;
  auto total_tracklength_l = total_tracklength_;
  auto in_flight = ptcls->get<3>();
  auto p_wgt = ptcls->get<4>();
  auto p_groups = ptcls->get<5>();
  auto xpoints_l = xpoints; // todo shouldn't need it, so remove

  auto evaluate_flux =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if ((mask > 0) && (in_flight(pid) == 1) && !ptcl_done[pid]) {
      OMEGA_H_CHECK_PRINTF(
          total_tracklength_l[pid] >= 0.0,
          "ERROR: Particle is moving but the tracklength is negative: %.16f\n",
          total_tracklength_l[pid]);

      Omega_h::Vector<3> dest = {xpoints_l[pid * 3 + 0], xpoints_l[pid * 3 + 1],
                                 xpoints_l[pid * 3 + 2]};
      Omega_h::Vector<3> orig = {prev_xpoint[pid * 3 + 0],
                                 prev_xpoint[pid * 3 + 1],
                                 prev_xpoint[pid * 3 + 2]};

      Omega_h::Real segment_length =
          Omega_h::norm(dest - orig); // / total_particles;
      if (segment_length >
          total_tracklength_l[pid] + 1e-6) { // tol for float operations and
                                             // search algorithm's inaccuracy
        // fixme: something wrong with orig: previous_xpoints are incorrect at
        // least for the first run
        printf("ERROR: Segment length in an element cannot be greater than the "
               "total tracklength but found %.16f, %.16f of pid %d crossing el "
               "%d starting at %d\nOrig: (%.16f, %.16f, %.16f), Dest: (%.16f, "
               "%.16f, %.16f)\n",
               segment_length, total_tracklength_l[pid], pid, elem_ids[pid], e,
               orig[0], orig[1], orig[2], dest[0], dest[1], dest[2]);
      }

      Omega_h::Real contribution = segment_length * p_wgt(pid);

      int group = p_groups(pid);
      OMEGA_H_CHECK_PRINTF(
          group >= 0 && group < flux.extent(1),
          "ERROR: Group %d is out of bounds for flux array with "
          "extent %lu\n",
          group, flux.extent(1));

      Kokkos::atomic_add(&flux(elem_ids[pid], group, 0), contribution);
      Kokkos::atomic_add(&flux(elem_ids[pid], group, 1),
                         contribution * contribution);
    }
  };
  pumipic::parallel_for(ptcls, evaluate_flux, "flux evaluation loop");
}

void PumiParticleAtElemBoundary::normalizeFlux(Omega_h::Mesh &mesh) {
  const Omega_h::LO nelems = mesh.nelems();
  const auto &el2n = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;
  const auto &coords = mesh.coords();

  auto flux = flux_;
  auto total_tracklength_l = total_tracklength_;

  Omega_h::Write<Omega_h::Real> tet_volumes(flux_.extent(0), -1.0,
                                            "tet_volumes");
  Omega_h::Write<Omega_h::Real> sd(flux_.extent(0), -1.0, "sd");

  auto normalize_flux_with_volume = OMEGA_H_LAMBDA(Omega_h::LO elem_id) {
    const auto elem_verts = Omega_h::gather_verts<4>(el2n, elem_id);
    const auto elem_vert_coords =
        Omega_h::gather_vectors<4, 3>(coords, elem_verts);

    auto b = Omega_h::simplex_basis<3, 3>(elem_vert_coords);
    auto volume = Omega_h::simplex_size_from_basis(b);

    tet_volumes[elem_id] = volume;
    for (int g = 0; g < flux.extent(1); g++) {
      flux(elem_id, g, 0) /= (volume * total_tracklength_l.size());
      flux(elem_id, g, 1) /= (volume * volume * total_tracklength_l.size());

      // calculate standard deviation FIXME this is not correct, needs number of
      // iterations
      flux(elem_id, g, 2) = Kokkos::sqrt(
          flux(elem_id, g, 1) - flux(elem_id, g, 0) * flux(elem_id, g, 0));
    }
  };
  Omega_h::parallel_for(tet_volumes.size(), normalize_flux_with_volume,
                        "normalize flux");

  mesh.add_tag(Omega_h::REGION, "volume", 1, Omega_h::Reals(tet_volumes));
}

void PumiParticleAtElemBoundary::finalizeAndWritePumiFlux(
    Omega_h::Mesh &full_mesh, const std::string &filename) {
  normalizeFlux(full_mesh);
  auto flux = flux_;
  int num_groups = flux.extent(1);
  int num_elems = flux.extent(0);
  Omega_h::Write<Omega_h::Real> normalized_flux(num_elems, 0.0,
                                                "normalized_flux");
  // copy flux to normalized_flux
  for (int g = 0; g < num_groups; g++) {
    auto copy_flux = OMEGA_H_LAMBDA(Omega_h::LO elem_id) {
      normalized_flux[elem_id] = flux(elem_id, g, 0);
    };
    Omega_h::parallel_for(num_elems, copy_flux, "copy flux to normalized_flux");
    std::string tag_name = "flux_group_" + std::to_string(g);
    Kokkos::fence();
    full_mesh.add_tag(Omega_h::REGION, tag_name, 1,
                      Omega_h::Reals(normalized_flux));
  }
  Omega_h::vtk::write_parallel(filename, &full_mesh, 3);
}

void pumiUpdatePtclPositions(PPPS *ptcls) {
  auto x_ps_d = ptcls->get<0>();
  auto xtgt_ps_d = ptcls->get<1>();
  auto updatePtclPos = PS_LAMBDA(const int &, const int &pid, const bool &) {
    x_ps_d(pid, 0) = xtgt_ps_d(pid, 0);
    x_ps_d(pid, 1) = xtgt_ps_d(pid, 1);
    x_ps_d(pid, 2) = xtgt_ps_d(pid, 2);
    xtgt_ps_d(pid, 0) = 0.0;
    xtgt_ps_d(pid, 1) = 0.0;
    xtgt_ps_d(pid, 2) = 0.0;
  };
  ps::parallel_for(ptcls, updatePtclPos);
}

void PumiParticleAtElemBoundary::compute_total_tracklength(PPPS *ptcls) {
  auto orig = ptcls->get<0>();
  auto dest = ptcls->get<1>();

  auto total_tracklength_l = total_tracklength_;

  auto computeTrackLength =
      PS_LAMBDA(const int &elemId, const int &pid, const bool &mask) {
    Omega_h::Vector<3> p_orig = {orig(pid, 0), orig(pid, 1), orig(pid, 2)};
    Omega_h::Vector<3> p_dest = {dest(pid, 0), dest(pid, 1), dest(pid, 2)};
    Omega_h::Real track_length = Omega_h::norm(p_dest - p_orig);
    total_tracklength_l[pid] = track_length;
  };
  pumipic::parallel_for(ptcls, computeTrackLength,
                        "compute total track length");
}

// search and update parent elements
//! @param initial initial search finds the initial location of the particles
//! and doesn't tally
void PumiTallyImpl::search_and_rebuild(bool initial, const bool migrate) {
  // initial cannot be false when is_pumipic_initialized is false
  // may fail if simulated more than one batch
  assert((is_pumipic_initialized == false && initial == true) ||
         (is_pumipic_initialized == true && initial == false));
  p_pumi_particle_at_elem_boundary_handler->mark_initial_as(initial);
  auto orig = pumipic_ptcls->get<0>();
  auto dest = pumipic_ptcls->get<1>();
  auto pid = pumipic_ptcls->get<2>();

  if (p_picparts_->mesh() == nullptr || p_picparts_->mesh()->nelems() == 0) {
    fprintf(stderr, "ERROR: Mesh is empty\n");
  }

  // total tracklengths are used to calculate the flux
  if (!initial) {
    p_pumi_particle_at_elem_boundary_handler->updatePrevXPoint(
        pumipic_ptcls.get());
    p_pumi_particle_at_elem_boundary_handler->compute_total_tracklength(
        pumipic_ptcls.get());
  }

  bool isFoundAll = p_particle_tracer_->search(migrate);

  if (!isFoundAll) {
    printf(
        "ERROR: Not all particles are found. May need more loops in search\n");
  }
}

std::unique_ptr<PPPS>
pp_create_particle_structure(Omega_h::Mesh mesh, pumipic::lid_t numPtcls,
                             pumiinopenmc::PPPS::kkLidView ptcls_per_elem) {
  Omega_h::Int ne = mesh.nelems();
  pumiinopenmc::PPPS::kkGidView element_gids("element_gids", ne);

  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy;

  Omega_h::parallel_for(
      ne, OMEGA_H_LAMBDA(const Omega_h::LO &i) { element_gids(i) = i; });

#ifdef PUMI_USE_KOKKOS_CUDA
  printf("PumiPIC Using GPU for simulation...\n");
  policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, 32);
#else
  printf("PumiPIC Using CPU for simulation...\n");
  policy =
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
#endif

  auto ptcls = std::make_unique<pumipic::DPS<pumiinopenmc::PPParticle>>(
      policy, ne, numPtcls, ptcls_per_elem, element_gids);

  return ptcls;
}

std::unique_ptr<PPPS> pp_create_particle_structure(Omega_h::Mesh mesh,
                                                   pumipic::lid_t numPtcls) {
  Omega_h::Int ne = mesh.nelems();
  pumiinopenmc::PPPS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  pumiinopenmc::PPPS::kkGidView element_gids("element_gids", ne);

  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy;

  Omega_h::parallel_for(
      ne, OMEGA_H_LAMBDA(const Omega_h::LO &i) { element_gids(i) = i; });

  Omega_h::parallel_for(
      mesh.nelems(), OMEGA_H_LAMBDA(Omega_h::LO id) {
        ptcls_per_elem[id] = (id == 0) ? numPtcls : 0;
      });

#ifdef PUMI_USE_KOKKOS_CUDA
  printf("PumiPIC Using GPU for simulation...\n");
  policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, 32);
#else
  printf("PumiPIC Using CPU for simulation...\n");
  policy =
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
#endif

  auto ptcls = std::make_unique<pumipic::DPS<pumiinopenmc::PPParticle>>(
      policy, ne, numPtcls, ptcls_per_elem, element_gids);

  return ptcls;
}

/*
void set_pumipic_particle_structure_size(int openmc_particles_in_flight, int
openmc_work_per_rank, int openmc_n_particles)
{
    int64_t n_particles; // TODO have a better way to do it than this
    // FIXME why this work_per rank is not set in the settings by now?


    if (openmc_particles_in_flight == 0 && openmc_work_per_rank == 0) {
        printf("While creating PumiPIC particle structure, both
max_particles_in_flight and work_per_rank are 0.\n"); n_particles =
(openmc_n_particles != 0) ? openmc_n_particles : pumi_ps_size; }else if
(openmc_particles_in_flight == 0 || openmc_work_per_rank == 0) { n_particles =
std::max(openmc_particles_in_flight, openmc_work_per_rank); printf("One of
max_particles_in_flight or work_per_rank is 0. Setting PumiPIC particle
structure size to %d\n", n_particles); } else { n_particles =
std::min(openmc_particles_in_flight, openmc_work_per_rank); printf("Setting
PumiPIC particle structure size to %d\n", n_particles);
    }

    pumi_ps_size = n_particles;
    printf("Creteating PumiPIC particle structure with size %d\n", n_particles);
}
*/

void start_pumi_particles_in_0th_element(Omega_h::Mesh &mesh,
                                         pumiinopenmc::PPPS *ptcls) {
  // find the centroid of the 0th element
  const auto &coords = mesh.coords();
  const auto &tet2node = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

  Omega_h::Write<Omega_h::Real> centroid_of_el0(3, 0.0, "centroid");

  auto find_centroid_of_el0 = OMEGA_H_LAMBDA(Omega_h::LO id) {
    const auto nodes = Omega_h::gather_verts<4>(tet2node, id);
    Omega_h::Few<Omega_h::Vector<3>, 4> tet_node_coords =
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
                        "set initial particle positions");
}

Omega_h::Mesh *PumiTallyImpl::partition_pumipic_mesh() {
  Omega_h::Write<Omega_h::LO> owners(full_mesh_.nelems(), 0, "owners");
  // all the particles are initialized in element 0 to do an initial search to
  // find the starting locations
  // of the openmc given particles.
  // p_picparts_ = new pumipic::Mesh(full_mesh_, Omega_h::LOs(owners));
  p_picparts_ =
      std::make_unique<pumipic::Mesh>(full_mesh_, Omega_h::LOs(owners));
  printf("PumiPIC mesh partitioned\n");
  Omega_h::Mesh *mesh = p_picparts_->mesh();
  return mesh;
}

void PumiTallyImpl::create_and_initialize_pumi_particle_structure(
    Omega_h::Mesh *mesh, SourceDistribution source_dist) {
  PPPS::kkLidView ppe("ptcls_per_elem", mesh->nelems());
  if (source_dist == SourceDistribution::UNIFORM) {
    distributeParticlesBasesOnVolume(*mesh, ppe, pumi_ps_size);
  }
  pumipic_ptcls =
      pp_create_particle_structure(*mesh, pumi_ps_size, ppe); // fixme
  if (source_dist == SourceDistribution::UNIFORM) {
    auto device_pos_buffer_l = device_pos_buffer_;
    initialize_uniform_source(*mesh, device_pos_buffer_l, ppe);
    // copy the device positions to the particle structure
    auto init_loc = pumipic_ptcls->get<0>();
    auto pids = pumipic_ptcls->get<2>();

    auto copy_initial_positions =
        PS_LAMBDA(const int &e, const int &pid, const int &mask) {
      if (mask > 0) {
        pids(pid) = pid;
        init_loc(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
        init_loc(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
        init_loc(pid, 2) = device_pos_buffer_l[pid * 3 + 2];
      }
    };
    pumipic::parallel_for(pumipic_ptcls.get(), copy_initial_positions,
                          "copy initial positions from device buffer");
  }
  if (source_dist == SourceDistribution::ZERO) {
    start_pumi_particles_in_0th_element(*mesh, pumipic_ptcls.get());
  }
  p_pumi_particle_at_elem_boundary_handler =
      std::make_unique<pumiinopenmc::PumiParticleAtElemBoundary>(
          mesh->nelems(), pumipic_ptcls->capacity());

  printf("PumiPIC Mesh and data structure created with %d and %d as particle "
         "structure capacity\n",
         p_picparts_->mesh()->nelems(), pumipic_ptcls->capacity());
}

void PumiTallyImpl::read_pumipic_lib_and_full_mesh(int &argc, char **&argv) {
  printf("Reading the Omega_h mesh %s to tally with tracklength estimator\n",
         oh_mesh_fname.c_str());
  init_pumi_libs(argc, argv);

  if (oh_mesh_fname.empty()) {
    printf("[ERROR] Omega_h mesh for PumiPIC is not given. Provide --ohMesh = "
           "<osh file>");
  }
  full_mesh_ = Omega_h::binary::read(oh_mesh_fname, &oh_lib);
  if (full_mesh_.dim() != 3) {
    printf("PumiPIC only works for 3D mesh now.\n");
  }
  OMEGA_H_CHECK_PRINTF(full_mesh_.has_tag(Omega_h::REGION, "class_id"),
                       "Mesh %s does not have class_id tag on regions.\n",
                       oh_mesh_fname.c_str());
  printf("PumiPIC Loaded mesh %s with %d elements\n", oh_mesh_fname.c_str(),
         full_mesh_.nelems());
}

void PumiTallyImpl::load_pumipic_mesh_and_init_particles(
    int &argc, char **&argv, SourceDistribution source_dist) {
  read_pumipic_lib_and_full_mesh(argc, argv);
  Omega_h::Mesh *mesh = partition_pumipic_mesh();
  create_and_initialize_pumi_particle_structure(mesh, source_dist);
  compute_boundary_normals(*mesh);
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

void distributeParticlesBasesOnVolume(Omega_h::Mesh &mesh,
                                      pumiinopenmc::PPPS::kkLidView ppe,
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
#ifdef DEBUG
    OMEGA_H_CHECK(area > 0.0);
#endif
    o::Real volume_fraction = vol / mesh_volume;
    ppe[e] = std::round(numPtcls * volume_fraction);
  };
  o::parallel_for(ne, distribute_based_on_volume);

  Omega_h::LO totPtcls = 0;
  Kokkos::parallel_reduce(
      ppe.size(),
      OMEGA_H_LAMBDA(const int i, Omega_h::LO &lsum) { lsum += ppe[i]; },
      totPtcls);

  // remove or add particles to match the total number of particles
  int extra_particles = numPtcls - totPtcls;
  // go throught the first extra_particles elements and add/remove one particle
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

OMEGA_H_DEVICE o::Few<o::Vector<3>, 3>
barycentric_basis(const o::Few<o::Vector<2>, 3> &tri_verts) {
  o::Few<o::Vector<3>, 3> basis;
  for (int i = 0; i < 3; i++) {
    basis[0][i] = tri_verts[i][0];
    basis[1][i] = tri_verts[i][1];
    basis[2][i] = 1.0;
  }
  return basis;
}

OMEGA_H_DEVICE o::Few<o::Vector<4>, 4>
barycentric_basis(const o::Few<o::Vector<3>, 4> &tet_verts) {
  o::Few<o::Vector<4>, 4> basis;
  for (int i = 0; i < 4; ++i) {
    basis[0][i] = tet_verts[i][0];
    basis[1][i] = tet_verts[i][1];
    basis[2][i] = tet_verts[i][2];
    basis[3][i] = 1.0;
  }
  return basis;
}

OMEGA_H_DEVICE o::Vector<3>
barycentric2real(const o::Few<o::Vector<3>, 4> &tet_verts,
                 const o::Vector<4> &bary) {
  o::Few<o::Vector<4>, 4> basis = barycentric_basis(tet_verts);
  o::Vector<3> real_coords;
  // real_coords = basis * bary;
  real_coords[0] = basis[0][0] * bary[0] + basis[1][0] * bary[1] +
                   basis[2][0] * bary[2] + basis[3][0] * bary[3];
  real_coords[1] = basis[0][1] * bary[0] + basis[1][1] * bary[1] +
                   basis[2][1] * bary[2] + basis[3][1] * bary[3];
  real_coords[2] = basis[0][2] * bary[0] + basis[1][2] * bary[1] +
                   basis[2][2] * bary[2] + basis[3][2] * bary[3];

  return real_coords;
}

void initialize_uniform_source(Omega_h::Mesh &mesh,
                               Omega_h::Write<Omega_h::Real> particle_positions,
                               pumiinopenmc::PPPS::kkLidView ppe) {
  int dim = mesh.dim();
  OMEGA_H_CHECK(dim == 3);

  // cumulative sum of particles per element
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

  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> random_pool;
  auto set_initial_positions = OMEGA_H_LAMBDA(const int &e) {
    auto pid_start = cumulative_particles[e];
    auto pid_end = cumulative_particles[e + 1];
    auto num_particles_in_element = pid_end - pid_start;
    OMEGA_H_CHECK(num_particles_in_element >= 0);

    for (Omega_h::LO pid = pid_start; pid < pid_end; ++pid) {
      auto gen = random_pool.get_state();
      o::Vector<4> random_bcc{0.0, 0.0, 0.0, 0.0};
      random_bcc[0] = gen.drand(0.0, 1.0);
      random_bcc[1] = gen.drand(0.0, 1.0);
      random_bcc[2] = gen.drand(0.0, 1.0);
      o::Real complimentary0 = 1.0 - random_bcc[0];
      o::Real complimentary1 = 1.0 - random_bcc[1];
      o::Real complimentary2 = 1.0 - random_bcc[2];

      bool more_than_one = random_bcc[0] + random_bcc[1] + random_bcc[2] > 1.0;
      random_bcc[0] = more_than_one ? complimentary0 : random_bcc[0];
      random_bcc[1] = more_than_one ? complimentary1 : random_bcc[1];
      random_bcc[2] = more_than_one ? complimentary2 : random_bcc[2];
      random_bcc[2] = 1.0 - random_bcc[0] - random_bcc[1] - random_bcc[2];
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

} // namespace pumiinopenmc
