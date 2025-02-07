//
// Created by Fuad Hasan on 12/3/24.
//

#include "pumipic_particle_data_structure.h"
#include <Omega_h_shape.hpp>
#include <pumipic_ptcl_ops.hpp>
#include <Omega_h_file.hpp>
#include <pumipic_library.hpp>
#include <pumipic_adjacency.hpp>
#include <pumipic_adjacency.tpp>
#include <Omega_h_mesh.hpp>
#include <pumipic_mesh.hpp>

#include <chrono>

namespace pumiinopenmc {
    struct TallyTimes {
        double initialization_time = 0.0;
        double total_time_to_tally = 0.0;
        double vtk_file_write_time = 0.0;

        void print_times() const {
            printf("\n");
            printf("[TIME] Initialization time     : %f seconds\n", initialization_time);
            printf("[TIME] Total time to tally     : %f seconds\n", total_time_to_tally);
            printf("[TIME] VTK file write time     : %f seconds\n", vtk_file_write_time);
            printf("[TIME] Total PumiPic time      : %f seconds\n",
                   initialization_time + total_time_to_tally + vtk_file_write_time);
        }
    };

    // ------------------------------------------------------------------------------------------------//
    // * Data structure for PumiPic
    // Particle: 0-origin, 1-destination, 2-particle_id, 3-in_advance_particle_queue, 4-weight
    typedef pumipic::MemberTypes<pumipic::Vector3d, pumipic::Vector3d, Omega_h::LO, Omega_h::I16, Omega_h::Real> PPParticle;
    typedef pumipic::ParticleStructure<PPParticle> PPPS;
    typedef Kokkos::DefaultExecutionSpace PPExeSpace;

    // ------------------------------------------------------------------------------------------------//
    // * Helper Functions
    void pp_move_to_new_element(Omega_h::Mesh &mesh, PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
                                Omega_h::Write<Omega_h::LO> &ptcl_done,
                                Omega_h::Write<Omega_h::LO> &lastExit);

    std::unique_ptr<PPPS> pp_create_particle_structure(Omega_h::Mesh mesh, pumipic::lid_t numPtcls);

    void start_pumi_particles_in_0th_element(Omega_h::Mesh &mesh, pumiinopenmc::PPPS *ptcls);


    PumiTally::~PumiTally() = default;

    // ------------------------------------------------------------------------------------------------//
    // * Struct for PumiParticleAtElemBoundary
    struct PumiParticleAtElemBoundary {
        PumiParticleAtElemBoundary(size_t nelems, size_t capacity);

        void operator()(Omega_h::Mesh &mesh, pumiinopenmc::PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
                        Omega_h::Write<Omega_h::LO> &inter_faces, Omega_h::Write<Omega_h::LO> &lastExit,
                        Omega_h::Write<Omega_h::Real> &inter_points, Omega_h::Write<Omega_h::LO> &ptcl_done);

        void updatePrevXPoint(Omega_h::Write<Omega_h::Real> &xpoints);

        void updatePrevXPoint(PPPS *ptcls);

        void evaluateFlux(PPPS *ptcls, Omega_h::Write<Omega_h::Real> xpoints, Omega_h::Write<Omega_h::LO> elem_ids,
                          Omega_h::Write<Omega_h::LO> ptcl_done);

        void finalizeAndWritePumiFlux(Omega_h::Mesh &full_mesh, const std::string &filename);

        Omega_h::Reals normalizeFlux(Omega_h::Mesh &mesh);

        void mark_initial_as(bool initial);

        void compute_total_tracklength(PPPS *ptcls);

        bool initial_; // in initial run, flux is not tallied
        Omega_h::Write<Omega_h::Real> flux_;
        Omega_h::Write<Omega_h::Real> prev_xpoint_;
        Omega_h::Write<Omega_h::Real> total_tracklength_;
    };
    // ------------------------------------------------------------------------------------------------//

    // ------------------------------------------------------------------------------------------------//
    // * Struct for PumiTallyImpl
    struct PumiTallyImpl {
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

        std::unique_ptr<PumiParticleAtElemBoundary> p_pumi_particle_at_elem_boundary_handler;

        Omega_h::Write<Omega_h::LO> elem_ids_;
        Omega_h::Write<Omega_h::Real> inter_points_;
        Omega_h::Write<Omega_h::LO> inter_faces_;

        Omega_h::Write<Omega_h::Real> device_pos_buffer_;
        Omega_h::Write<Omega_h::I8> device_in_adv_que_;
        Omega_h::Write<Omega_h::Real> weights_;

        TallyTimes tally_times;

        // * Constructor
        PumiTallyImpl(std::string &mesh_filename, int64_t num_particles, int &argc, char **&argv); // fixme extra &

        // * Destructor
        ~PumiTallyImpl() = default;

        // Functions
        void create_and_initialize_pumi_particle_structure(Omega_h::Mesh *mesh);

        void load_pumipic_mesh_and_init_particles(int &argc, char **&argv);

        Omega_h::Mesh *partition_pumipic_mesh();

        void init_pumi_libs(int &argc, char **&argv);

        void search_and_rebuild(bool initial, bool migrate = true);

        void read_pumipic_lib_and_full_mesh(int &argc, char **&argv);

        void initialize_particle_location(double *init_particle_positions, int64_t size);

        void move_to_next_location(double *particle_destinations, int8_t *flying, double *weights, int64_t size);

        void write_pumi_tally_mesh();

        void copy_data_to_device(double *init_particle_positions);

        void search_initial_elements();

        void copy_and_reset_flying_flag(int8_t *flying);

        void copy_weights(double *weights);
    };

    PumiTallyImpl::PumiTallyImpl(std::string &mesh_filename, int64_t num_particles, int &argc, char **&argv) {
        pumi_ps_size = num_particles;
        oh_mesh_fname = mesh_filename;

        elem_ids_ = Omega_h::Write<Omega_h::LO>(pumi_ps_size, 0, "element_ids");

        device_pos_buffer_ = Omega_h::Write<Omega_h::Real>(pumi_ps_size * 3, 0.0, "device_pos_buffer");
        device_in_adv_que_ = Omega_h::Write<Omega_h::I8>(pumi_ps_size, 0, "device_in_adv_que");
        weights_           = Omega_h::Write<Omega_h::Real>(pumi_ps_size, 0.0, "weights");

        // todo can track lengths be here?

        load_pumipic_mesh_and_init_particles(argc, argv);
        start_pumi_particles_in_0th_element(*p_picparts_->mesh(), pumipic_ptcls.get());
    }

    void PumiTallyImpl::initialize_particle_location(double *init_particle_positions, int64_t size) {
        // copy to host buffer
        assert(size == pumi_ps_size * 3);
        copy_data_to_device(init_particle_positions);
        search_initial_elements();
        // TODO Get total initial particle weight
#ifdef PUMI_MEASURE_TIME
        Kokkos::fence();
#endif
    }

    void
    PumiTallyImpl::move_to_next_location(double *particle_destinations, int8_t *flying, double *weights, int64_t size) {
        assert(size == pumi_ps_size * 3);

        // copy to device buffer
        copy_data_to_device(particle_destinations);
        // copy fly to device buffer
        copy_and_reset_flying_flag(flying);
        copy_weights(weights);

        // copy position buffer ps
        auto particle_dest = pumipic_ptcls->get<1>();
        auto in_flight = pumipic_ptcls->get<3>();

        int64_t pumi_ps_size_ = pumi_ps_size;
        const auto &device_pos_buffer_l = device_pos_buffer_;
        const auto &device_in_adv_que_l = device_in_adv_que_;

        auto set_particle_dest = PS_LAMBDA(const int &e, const int &pid, const int &mask) {
            if (mask > 0 && pid < pumi_ps_size_) {
                particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
                particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
                particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];

                // everyone is in flight for this initial search
                in_flight(pid) = device_in_adv_que_l[pid];
            }
        };
        pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest, "set particle position as dest");

        bool migrate = iter_count_ % 100 == 0;
        iter_count_++;
        search_and_rebuild(false, migrate);
#ifdef PUMI_MEASURE_TIME
        Kokkos::fence();
#endif
    }

    void PumiTallyImpl::write_pumi_tally_mesh() {
        p_pumi_particle_at_elem_boundary_handler->finalizeAndWritePumiFlux(full_mesh_, "fluxresult.vtk");
#ifdef PUMI_MEASURE_TIME
        Kokkos::fence();
#endif
    }

    void PumiTallyImpl::copy_and_reset_flying_flag(int8_t *flying) {
        // todo get the size too
        auto device_in_adv_que_l = device_in_adv_que_;
        Kokkos::View<Omega_h::I8 *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                host_flying_view(flying, pumi_ps_size);
        Kokkos::View<Omega_h::I8 *, PPExeSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                device_flying_view(device_in_adv_que_l.data(), device_in_adv_que_l.size());
        Kokkos::deep_copy(device_flying_view, host_flying_view);

        for (int64_t pid = 0; pid < pumi_ps_size; ++pid) {
            // reset flying flag to zero
            flying[pid] = 0;
        }
    }

    void PumiTallyImpl::copy_weights(double *weights) {
        auto weights_l = weights_;
        Kokkos::View<Omega_h::Real *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        host_weights_view(weights, pumi_ps_size);
        Kokkos::View<Omega_h::Real *, PPExeSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        device_weights_view(weights_l.data(), weights_l.size());

        Kokkos::deep_copy(device_weights_view, host_weights_view);

        auto p_wgt = pumipic_ptcls->get<4>();
        auto copy_particle_weights = PS_LAMBDA(
                const int &e,
                const int &pid,
                const int &mask) {
            p_wgt(pid) = weights_l[pid];
        };
        pumipic::parallel_for(pumipic_ptcls.get(), copy_particle_weights, "copy particle weights");
    }

    void PumiTallyImpl::search_initial_elements() {// assign the location to ptcl dest
        auto particle_dest = pumipic_ptcls->get<1>();
        auto in_flight = pumipic_ptcls->get<3>();

        int64_t pumi_ps_size_l = pumi_ps_size;
        const auto &device_pos_buffer_l = device_pos_buffer_;

        auto set_particle_dest = PS_LAMBDA(const int &e, const int &pid, const int &mask) {
            if (mask > 0 && pid < pumi_ps_size_l) {
                particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
                particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
                particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];

                // everyone is in flight for this initial search
                in_flight(pid) = 1;
            }
        };
        pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest, "set initial position as dest");

        // *initial* build and search to find the initial elements of the particles
        search_and_rebuild(true, true);
        is_pumipic_initialized = true;
    }

    void PumiTallyImpl::copy_data_to_device(double *init_particle_positions) {
        // fixme it should get size too to avoid memory error
        Kokkos::View<Omega_h::Real *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                host_pos_view(init_particle_positions, pumi_ps_size * 3);

        Kokkos::View<Omega_h::Real *, PPExeSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                device_pos_view(device_pos_buffer_.data(), pumi_ps_size * 3);

        Kokkos::deep_copy(device_pos_view, host_pos_view);
    }


    // methods for PumiTallyImpl and PumiParticleAtElemBoundary

    void PumiTallyImpl::init_pumi_libs(int &argc, char **&argv) {
        pp_lib = std::make_unique<pumipic::Library>(&argc, &argv);
        oh_lib = pp_lib->omega_h_lib();
    }

    void pp_move_to_new_element(Omega_h::Mesh &mesh, PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
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

    void apply_boundary_condition(Omega_h::Mesh &mesh, PPPS *ptcls,
                                  Omega_h::Write<Omega_h::LO> &elem_ids,
                                  Omega_h::Write<Omega_h::LO> &ptcl_done,
                                  Omega_h::Write<Omega_h::LO> &lastExit,
                                  Omega_h::Write<Omega_h::LO> &xFace,
                                  Omega_h::Write<Omega_h::Real> &inter_points) {

        // TODO: make this a member variable of the struct
        const auto &side_is_exposed = Omega_h::mark_exposed_sides(&mesh);
        auto particle_destination = ptcls->get<1>();
        auto checkExposedEdges =
                PS_LAMBDA(const int e, const int pid, const int mask) {
                    if (mask > 0 && !ptcl_done[pid]) {
                        assert(lastExit[pid] != -1);
                        const Omega_h::LO bridge = lastExit[pid];
                        const bool exposed = side_is_exposed[bridge];
                        ptcl_done[pid] = exposed;
                        xFace[pid] = lastExit[pid];
                        particle_destination(pid, 0) = (exposed) ? inter_points[pid*3] : particle_destination(pid, 0);
                        particle_destination(pid, 1) = (exposed) ? inter_points[pid*3+1] : particle_destination(pid, 1);
                        particle_destination(pid, 2) = (exposed) ? inter_points[pid*3+2] : particle_destination(pid, 2);
                        //elem_ids[pid] = exposed ? -1 : elem_ids[pid];
                    }
                };
        pumipic::parallel_for(ptcls, checkExposedEdges, "apply vacumm boundary condition");
    }

    PumiParticleAtElemBoundary::PumiParticleAtElemBoundary(size_t nelems, size_t capacity)
            : flux_(nelems, 0.0, "flux"),
              prev_xpoint_(capacity * 3, 0.0, "prev_xpoint"),
              total_tracklength_(capacity, 0.0, "total_tracklength"),
              initial_(true) {
        printf(
                "[INFO] Particle handler at boundary with %d elements and %d "
                "x points size (3 * n_particles)\n",
                flux_.size(), prev_xpoint_.size());
    }

    void PumiParticleAtElemBoundary::operator()(Omega_h::Mesh &mesh, pumiinopenmc::PPPS *ptcls,
                                                Omega_h::Write<Omega_h::LO> &elem_ids,
                                                Omega_h::Write<Omega_h::LO> &inter_faces,
                                                Omega_h::Write<Omega_h::LO> &lastExit,
                                                Omega_h::Write<Omega_h::Real> &inter_points,
                                                Omega_h::Write<Omega_h::LO> &ptcl_done) {
        if (!initial_) {
            evaluateFlux(ptcls, inter_points, elem_ids, ptcl_done);
            updatePrevXPoint(inter_points);
        }
        apply_boundary_condition(mesh, ptcls, elem_ids, ptcl_done, lastExit, inter_faces, inter_points);
        pp_move_to_new_element(mesh, ptcls, elem_ids, ptcl_done, lastExit);
    }

    void PumiParticleAtElemBoundary::mark_initial_as(bool initial) {
        initial_ = initial;
    }

    void PumiParticleAtElemBoundary::updatePrevXPoint(Omega_h::Write<Omega_h::Real> &xpoints) {
        OMEGA_H_CHECK_PRINTF(
                xpoints.size() <= prev_xpoint_.size() && prev_xpoint_.size() != 0,
                "xpoints size %d is greater than prev_xpoint size %d\n", xpoints.size(),
                prev_xpoint_.size());
        auto &prev_xpoint = prev_xpoint_;
        auto update = OMEGA_H_LAMBDA(Omega_h::LO
                                     i) { prev_xpoint[i] = xpoints[i]; };
        Omega_h::parallel_for(xpoints.size(), update, "update previous xpoints");
    }

    void PumiParticleAtElemBoundary::updatePrevXPoint(PPPS *ptcls) {
        // todo add checks of size
        auto prev_xpoints_l = prev_xpoint_;
        OMEGA_H_CHECK_PRINTF(ptcls->capacity() * 3 == prev_xpoints_l.size(),
                             "Error: prev_xpoints_s are not size properly capacity %d size %d\n", ptcls->capacity(),
                             prev_xpoints_l.size());
        auto xpoints = ptcls->get<0>();
        auto update = PS_LAMBDA(
                const auto &e,
                const auto &pid,
                const auto &mask) {
            prev_xpoints_l[pid * 3 + 0] = xpoints(pid, 0);
            prev_xpoints_l[pid * 3 + 1] = xpoints(pid, 1);
            prev_xpoints_l[pid * 3 + 2] = xpoints(pid, 2);
        };
        pumipic::parallel_for(ptcls, update, "update previous xpoints from origin points");
    }

    void PumiParticleAtElemBoundary::evaluateFlux(PPPS *ptcls, Omega_h::Write<Omega_h::Real> xpoints,
                                                  Omega_h::Write<Omega_h::LO> elem_ids,
                                                  Omega_h::Write<Omega_h::LO> ptcl_done) {
        //Omega_h::Real total_particles = ptcls->nPtcls();
        auto prev_xpoint = prev_xpoint_;
        auto flux = flux_;
        auto total_tracklength_l = total_tracklength_;
        auto in_flight = ptcls->get<3>();
        auto p_wgt = ptcls->get<4>();
        auto xpoints_l = xpoints; // todo shouldn't need it, so remove

        auto evaluate_flux =
                PS_LAMBDA(
                        const int &e,
                        const int &pid,
                        const int &mask) {
                    if (mask > 0 && in_flight(pid) == 1 && !ptcl_done[pid]) {
                        OMEGA_H_CHECK_PRINTF(total_tracklength_l[pid] >= 0.0,
                                             "ERROR: Particle is moving but the tracklength is negative: %.16f\n",
                                             total_tracklength_l[pid]);

                        Omega_h::Vector<3> dest = {xpoints_l[pid * 3 + 0],
                                                   xpoints_l[pid * 3 + 1],
                                                   xpoints_l[pid * 3 + 2]};
                        Omega_h::Vector<3> orig = {prev_xpoint[pid * 3 + 0],
                                                   prev_xpoint[pid * 3 + 1],
                                                   prev_xpoint[pid * 3 + 2]};

                        Omega_h::Real segment_length = Omega_h::norm(dest - orig);  // / total_particles;
                        if (segment_length > total_tracklength_l[pid] +
                                             1e-6) { // tol for float operations and search algorithm's inaccuracy
                            // fixme: something wrong with orig: previous_xpoints are incorrect at least for the first run
                            printf("ERROR: Segment length in an element cannot be greater than the total tracklength but found %.16f, %.16f of pid %d crossing el %d starting at %d\nOrig: (%.16f, %.16f, %.16f), Dest: (%.16f, %.16f, %.16f)\n",
                                   segment_length, total_tracklength_l[pid], pid, elem_ids[pid], e,
                                   orig[0], orig[1], orig[2], dest[0], dest[1], dest[2]);
                        }

                        Omega_h::Real contribution = segment_length * p_wgt(pid);

                        Kokkos::atomic_add(&flux[elem_ids[pid]], contribution);
                    }
                };
        pumipic::parallel_for(ptcls, evaluate_flux, "flux evaluation loop");
    }

    Omega_h::Reals PumiParticleAtElemBoundary::normalizeFlux(Omega_h::Mesh &mesh) {
        const Omega_h::LO nelems = mesh.nelems();
        const auto &el2n = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;
        const auto &coords = mesh.coords();

        auto flux = flux_;
        auto total_tracklength_l = total_tracklength_;

        Omega_h::Write<Omega_h::Real> tet_volumes(flux_.size(), -1.0, "tet_volumes");
        Omega_h::Write<Omega_h::Real> normalized_flux(flux_.size(), -1.0, "normalized flux");

        auto normalize_flux_with_volume = OMEGA_H_LAMBDA(Omega_h::LO
                                                         elem_id) {
            const auto elem_verts = Omega_h::gather_verts<4>(el2n, elem_id);
            const auto elem_vert_coords = Omega_h::gather_vectors<4, 3>(coords, elem_verts);

            auto b = Omega_h::simplex_basis<3, 3>(elem_vert_coords);
            auto volume = Omega_h::simplex_size_from_basis(b);

            tet_volumes[elem_id] = volume;
            normalized_flux[elem_id] = flux[elem_id] / (volume * total_tracklength_l.size());
        };
        Omega_h::parallel_for(tet_volumes.size(), normalize_flux_with_volume,
                              "normalize flux");

        mesh.add_tag(Omega_h::REGION, "volume", 1, Omega_h::Reals(tet_volumes));
        return {normalized_flux};
    }

    void PumiParticleAtElemBoundary::finalizeAndWritePumiFlux(Omega_h::Mesh &full_mesh, const std::string &filename) {
        const auto &normalized_flux = normalizeFlux(full_mesh);
        full_mesh.add_tag(Omega_h::REGION, "flux", 1, normalized_flux);
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

    void pumiRebuild(pumipic::Mesh *picparts, PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
                     const bool migrate = true) {
        pumiUpdatePtclPositions(ptcls);
        if (migrate) {
            pumipic::migrate_lb_ptcls(*picparts, ptcls, elem_ids, 1.05);
            pumipic::printPtclImb(ptcls);
        }
    }

    void PumiParticleAtElemBoundary::compute_total_tracklength(PPPS *ptcls) {
        auto orig = ptcls->get<0>();
        auto dest = ptcls->get<1>();

        auto total_tracklength_l = total_tracklength_;

        auto computeTrackLength = PS_LAMBDA(
                const int &elemId,
                const int &pid,
                const bool &mask) {
            Omega_h::Vector<3> p_orig = {orig(pid, 0), orig(pid, 1), orig(pid, 2)};
            Omega_h::Vector<3> p_dest = {dest(pid, 0), dest(pid, 1), dest(pid, 2)};
            Omega_h::Real track_length = Omega_h::norm(p_dest - p_orig);
            total_tracklength_l[pid] = track_length;
        };
        pumipic::parallel_for(ptcls, computeTrackLength, "compute total track length");
    }

    // search and update parent elements
    //! @param initial initial search finds the initial location of the particles and doesn't tally
    void PumiTallyImpl::search_and_rebuild(bool initial, const bool migrate) {
        // initial cannot be false when is_pumipic_initialized is false
        // may fail if simulated more than one batch
        assert((is_pumipic_initialized == false && initial == true) ||
               (is_pumipic_initialized == true && initial == false));
        p_pumi_particle_at_elem_boundary_handler->mark_initial_as(initial);
        Omega_h::LO maxLoops = 1000;
        auto orig = pumipic_ptcls->get<0>();
        auto dest = pumipic_ptcls->get<1>();
        auto pid = pumipic_ptcls->get<2>();

        if (p_picparts_->mesh() == nullptr || p_picparts_->mesh()->nelems() == 0) {
            fprintf(stderr, "ERROR: Mesh is empty\n");
        }

        // total tracklengths are used to calculate the flux
        if (!initial) {
            p_pumi_particle_at_elem_boundary_handler->updatePrevXPoint(pumipic_ptcls.get());
            p_pumi_particle_at_elem_boundary_handler->compute_total_tracklength(pumipic_ptcls.get());
        }

        auto elem_ids_l = elem_ids_;
        auto inter_faces_l = inter_faces_;
        auto inter_points_l = inter_points_;

        bool isFoundAll = pumipic::particle_search(*p_picparts_->mesh(), pumipic_ptcls.get(),
                                                   orig, dest, pid, elem_ids_l, inter_faces_l,
                                                   inter_points_l, maxLoops, *p_pumi_particle_at_elem_boundary_handler);
        if (!isFoundAll) {
            printf("ERROR: Not all particles are found. May need more loops in search\n");
        }

        pumiRebuild(p_picparts_.get(), pumipic_ptcls.get(), elem_ids_l, migrate);
    }


    std::unique_ptr<PPPS> pp_create_particle_structure(Omega_h::Mesh mesh, pumipic::lid_t numPtcls) {
        Omega_h::Int ne = mesh.nelems();
        pumiinopenmc::PPPS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
        pumiinopenmc::PPPS::kkGidView element_gids("element_gids", ne);

        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy;

        Omega_h::parallel_for(
                ne, OMEGA_H_LAMBDA(
                        const Omega_h::LO &i) { element_gids(i) = i; });

        Omega_h::parallel_for(mesh.nelems(),
                              OMEGA_H_LAMBDA(Omega_h::LO
                                             id) {
                                  ptcls_per_elem[id] = (id == 0) ? numPtcls : 0;
                              });

#ifdef PUMI_USE_KOKKOS_CUDA
        printf("PumiPIC Using GPU for simulation...\n");
        policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, 32);
#else
        printf("PumiPIC Using CPU for simulation...\n");
        policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
#endif

        auto ptcls = std::make_unique<pumipic::DPS<pumiinopenmc::PPParticle>>(policy, ne, numPtcls,
                                                                              ptcls_per_elem, element_gids);

        return ptcls;
    }

    /*
    void set_pumipic_particle_structure_size(int openmc_particles_in_flight, int openmc_work_per_rank, int openmc_n_particles)
    {
        int64_t n_particles; // TODO have a better way to do it than this
        // FIXME why this work_per rank is not set in the settings by now?


        if (openmc_particles_in_flight == 0 && openmc_work_per_rank == 0) {
            printf("While creating PumiPIC particle structure, both max_particles_in_flight and work_per_rank are 0.\n");
            n_particles = (openmc_n_particles != 0) ? openmc_n_particles : pumi_ps_size;
        }else if (openmc_particles_in_flight == 0 || openmc_work_per_rank == 0) {
            n_particles = std::max(openmc_particles_in_flight, openmc_work_per_rank);
            printf("One of max_particles_in_flight or work_per_rank is 0. Setting PumiPIC particle structure size to %d\n", n_particles);
        } else {
            n_particles = std::min(openmc_particles_in_flight, openmc_work_per_rank);
            printf("Setting PumiPIC particle structure size to %d\n", n_particles);
        }

        pumi_ps_size = n_particles;
        printf("Creteating PumiPIC particle structure with size %d\n", n_particles);
    }
    */

    void start_pumi_particles_in_0th_element(Omega_h::Mesh &mesh, pumiinopenmc::PPPS *ptcls) {
        // find the centroid of the 0th element
        const auto &coords = mesh.coords();
        const auto &tet2node = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

        Omega_h::Write<Omega_h::Real> centroid_of_el0(3, 0.0, "centroid");

        auto find_centroid_of_el0 = OMEGA_H_LAMBDA(Omega_h::LO
                                                   id) {
            const auto nodes = Omega_h::gather_verts<4>(tet2node, id);
            Omega_h::Few<Omega_h::Vector<3>, 4> tet_node_coords = Omega_h::gather_vectors<4, 3>(coords, nodes);
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

        auto set_initial_positions = PS_LAMBDA(const int &e, const int &pid, const int &mask) {
            if (mask > 0) {
                pids(pid) = pid;
                in_fly(pid) = 1;
                init_loc(pid, 0) = centroid_of_el0[0];
                init_loc(pid, 1) = centroid_of_el0[1];
                init_loc(pid, 2) = centroid_of_el0[2];
            }
        };
        pumipic::parallel_for(ptcls, set_initial_positions, "set initial particle positions");
    }

    Omega_h::Mesh *PumiTallyImpl::partition_pumipic_mesh() {
        Omega_h::Write<Omega_h::LO> owners(full_mesh_.nelems(), 0, "owners");
        // all the particles are initialized in element 0 to do an initial search to
        // find the starting locations
        // of the openmc given particles.
        //p_picparts_ = new pumipic::Mesh(full_mesh_, Omega_h::LOs(owners));
        p_picparts_ = std::make_unique<pumipic::Mesh>(full_mesh_, Omega_h::LOs(owners));
        printf("PumiPIC mesh partitioned\n");
        Omega_h::Mesh *mesh = p_picparts_->mesh();
        return mesh;
    }

    void PumiTallyImpl::create_and_initialize_pumi_particle_structure(Omega_h::Mesh *mesh) {
        pumipic_ptcls = pp_create_particle_structure(*mesh, pumi_ps_size);
        start_pumi_particles_in_0th_element(*mesh, pumipic_ptcls.get());
        p_pumi_particle_at_elem_boundary_handler =
                std::make_unique<pumiinopenmc::PumiParticleAtElemBoundary>(mesh->nelems(),
                                                                           pumipic_ptcls->capacity());

        printf("PumiPIC Mesh and data structure created with %d and %d as particle structure capacity\n",
               p_picparts_->mesh()->nelems(), pumipic_ptcls->capacity());
    }

    void PumiTallyImpl::read_pumipic_lib_and_full_mesh(int &argc, char **&argv) {
        printf("Reading the Omega_h mesh %s to tally with tracklength estimator\n", oh_mesh_fname.c_str());
        init_pumi_libs(argc, argv);

        if (oh_mesh_fname.empty()) {
            printf("[ERROR] Omega_h mesh for PumiPIC is not given. Provide --ohMesh = <osh file>");
        }
        full_mesh_ = Omega_h::binary::read(oh_mesh_fname, &oh_lib);
        if (full_mesh_.dim() != 3) {
            printf("PumiPIC only works for 3D mesh now.\n");
        }
        printf("PumiPIC Loaded mesh %s with %d elements\n", oh_mesh_fname.c_str(), full_mesh_.nelems());
    }

    void PumiTallyImpl::load_pumipic_mesh_and_init_particles(int &argc, char **&argv) {
        read_pumipic_lib_and_full_mesh(argc, argv);
        Omega_h::Mesh *mesh = partition_pumipic_mesh();
        create_and_initialize_pumi_particle_structure(mesh);
    }

    PumiTally::PumiTally(std::string &mesh_filename, int64_t num_particles, int &argc, char **&argv)
            : pimpl(std::make_unique<PumiTallyImpl>(mesh_filename, num_particles, argc, argv)) {
    }

    void PumiTally::initialize_particle_location(double *init_particle_positions, int64_t size) {
        auto start_time = std::chrono::steady_clock::now();

        pimpl->initialize_particle_location(init_particle_positions, size);

        std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start_time;
        pimpl->tally_times.initialization_time += elapsed_seconds.count();
    }

    void
    PumiTally::move_to_next_location(double *particle_destinations, int8_t *flying, double *weights, int64_t size) {
        auto start_time = std::chrono::steady_clock::now();

        pimpl->move_to_next_location(particle_destinations, flying, weights, size);

        std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start_time;
        pimpl->tally_times.total_time_to_tally += elapsed_seconds.count();
    }

    void PumiTally::write_pumi_tally_mesh() {
        auto start_time = std::chrono::steady_clock::now();

        pimpl->write_pumi_tally_mesh();

        std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start_time;
        pimpl->tally_times.vtk_file_write_time += elapsed_seconds.count();
        pimpl->tally_times.print_times();
    }

} // namespace pumiinopenmc
