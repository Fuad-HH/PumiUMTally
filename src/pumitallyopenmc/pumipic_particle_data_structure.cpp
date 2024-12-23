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
    struct TallyTimes{
        double initialization_time = 0.0;
        double total_time_to_tally = 0.0;
        double vtk_file_write_time = 0.0;

        void print_times() const{
            printf("\n");
            printf("[TIME] Initialization time     : %f seconds\n", initialization_time);
            printf("[TIME] Total time to tally     : %f seconds\n", total_time_to_tally);
            printf("[TIME] VTK file write time     : %f seconds\n", vtk_file_write_time);
            printf("[TIME] Total PumiPic time      : %f seconds\n", initialization_time + total_time_to_tally + vtk_file_write_time);
        }
    };

    // ------------------------------------------------------------------------------------------------//
    // * Data structure for PumiPic
    // Particle: origin, destination, particle_id, in_advance_particle_queue
    typedef pumipic::MemberTypes<pumipic::Vector3d, pumipic::Vector3d, Omega_h::LO, Omega_h::I16> PPParticle;
    typedef pumipic::ParticleStructure<PPParticle> PPPS;
    typedef Kokkos::DefaultExecutionSpace PPExeSpace;

    // ------------------------------------------------------------------------------------------------//
    // * Helper Functions
    void pp_move_to_new_element(Omega_h::Mesh &mesh, PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
                                Omega_h::Write<Omega_h::LO> &ptcl_done,
                                Omega_h::Write<Omega_h::LO> &lastExit);
    std::unique_ptr<PPPS> pp_create_particle_structure(Omega_h::Mesh mesh, pumipic::lid_t numPtcls);
    void start_pumi_particles_in_0th_element(Omega_h::Mesh& mesh, pumiinopenmc::PPPS* ptcls);


    PumiTally::~PumiTally() = default;

    // ------------------------------------------------------------------------------------------------//
    // * Struct for PumiParticleAtElemBoundary
    struct PumiParticleAtElemBoundary {
        PumiParticleAtElemBoundary(size_t nelems, size_t capacity);

        void operator()(Omega_h::Mesh &mesh, pumiinopenmc::PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
                        Omega_h::Write<Omega_h::LO> &inter_faces, Omega_h::Write<Omega_h::LO> &lastExit,
                        Omega_h::Write<Omega_h::Real> &inter_points, Omega_h::Write<Omega_h::LO> &ptcl_done);

        void updatePrevXPoint(Omega_h::Write<Omega_h::Real> &xpoints);

        void evaluateFlux(PPPS *ptcls, Omega_h::Write<Omega_h::Real> &xpoints);
        void finalizeAndWritePumiFlux(Omega_h::Mesh& full_mesh, const std::string& filename);
        Omega_h::Reals normalizeFlux(Omega_h::Mesh &mesh);

        void mark_initial_as(bool initial);

        bool initial_; // in initial run, flux is not tallied
        Omega_h::Write<Omega_h::Real> flux_;
        Omega_h::Write<Omega_h::Real> prev_xpoint_;
    };
    // ------------------------------------------------------------------------------------------------//

    // ------------------------------------------------------------------------------------------------//
    // * Struct for PumiTallyImpl
    struct PumiTallyImpl {
        int64_t pumi_ps_size        = 1000000; // hundred thousand
        std::string oh_mesh_fname;

        Omega_h::Library oh_lib;
        Omega_h::Mesh full_mesh_;

        std::unique_ptr<pumipic::Library> pp_lib = nullptr;
        std::unique_ptr<pumipic::Mesh> p_picparts_ = nullptr;
        std::unique_ptr<PPPS> pumipic_ptcls = nullptr;

        long double pumipic_tol     = 1e-8;
        bool is_pumipic_initialized = false;

        std::unique_ptr<PumiParticleAtElemBoundary> p_pumi_particle_at_elem_boundary_handler;

        Omega_h::Write<Omega_h::LO> elem_ids_;
        Omega_h::Write<Omega_h::Real> inter_points_;
        Omega_h::Write<Omega_h::LO> inter_faces_;

        Omega_h::HostWrite<Omega_h::Real> host_pos_buffer_;
        Omega_h::Write<Omega_h::Real> device_pos_buffer_;
        Omega_h::HostWrite<Omega_h::I8> host_in_adv_que_;
        Omega_h::Write<Omega_h::I8> device_in_adv_que_;

        TallyTimes tally_times;

        // * Constructor
        PumiTallyImpl(std::string& mesh_filename, int64_t num_particles, int& argc, char**& argv);

        // * Destructor
        ~PumiTallyImpl() {
            Kokkos::finalize();
        }

        // Functions
        void create_and_initialize_pumi_particle_structure(Omega_h::Mesh* mesh);
        void load_pumipic_mesh_and_init_particles(int& argc, char**& argv);
        Omega_h::Mesh* partition_pumipic_mesh();
        void init_pumi_libs(int &argc, char **&argv);
        void search_and_rebuild(bool initial);
        void read_pumipic_lib_and_full_mesh(int& argc, char**& argv);
        void initialize_particle_location(double* init_particle_positions, int64_t size);
        void move_to_next_location(double* particle_destinations, int8_t* flying, int64_t num_particles);
        void write_pumi_tally_mesh();
        [[maybe_unused]] void copy_to_device_position_buffer(const double *init_particle_positions);
        void copy_data_to_device(double *init_particle_positions);
        void search_initial_elements();

        void copy_flying_flag(const int8_t *flying);
    };

    PumiTallyImpl::PumiTallyImpl(std::string &mesh_filename, int64_t num_particles, int &argc, char **&argv) {
        pumi_ps_size = num_particles;
        oh_mesh_fname = mesh_filename;

        host_pos_buffer_    = Omega_h::HostWrite    <Omega_h::Real> (pumi_ps_size * 3, 0.0, 0, "host_pos_buffer");
        device_pos_buffer_  = Omega_h::Write        <Omega_h::Real> (pumi_ps_size * 3, 0.0, "device_pos_buffer");
        // flies if 1, 0 if not (default doesn't fly)
        host_in_adv_que_    = Omega_h::HostWrite    <Omega_h::I8>   (pumi_ps_size, 0, 0, "host_in_adv_que");
        device_in_adv_que_  = Omega_h::Write        <Omega_h::I8>   (pumi_ps_size, 0, "device_in_adv_que");

        load_pumipic_mesh_and_init_particles(argc, argv);
        start_pumi_particles_in_0th_element(*p_picparts_->mesh(), pumipic_ptcls.get());
    }

    void PumiTallyImpl::initialize_particle_location(double *init_particle_positions, int64_t size) {
        // copy to host buffer
        assert(size == pumi_ps_size*3);
        copy_data_to_device(init_particle_positions);
        search_initial_elements();
    }

    void PumiTallyImpl::move_to_next_location(double *particle_destinations, int8_t *flying, int64_t size) {
        assert(size == pumi_ps_size*3);

        // copy to device buffer
        copy_data_to_device(particle_destinations);
        // copy fly to device buffer
        copy_flying_flag(flying);

        // copy position buffer ps
        auto particle_dest = pumipic_ptcls->get<1>();
        auto in_flight     = pumipic_ptcls->get<3>();

        int64_t pumi_ps_size_ = pumi_ps_size;
        const auto& device_pos_buffer_l = device_pos_buffer_;
        const auto& device_in_adv_que_l = device_in_adv_que_;

        auto set_particle_dest = PS_LAMBDA(const int &e, const int &pid, const int &mask){
            if (mask>0 && pid < pumi_ps_size_){
                particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
                particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
                particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];

                // everyone is in flight for this initial search
                in_flight(pid) = device_in_adv_que_l[pid];
            }
        };
        pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest, "set particle position as dest");

        search_and_rebuild(false);
    }

    void PumiTallyImpl::write_pumi_tally_mesh() {
        p_pumi_particle_at_elem_boundary_handler->finalizeAndWritePumiFlux(full_mesh_, "fluxresult.vtk");
    }

    void PumiTallyImpl::copy_flying_flag(const int8_t *flying) {
        for (int64_t pid = 0; pid < pumi_ps_size; pid++) {
            host_in_adv_que_[pid] = flying[pid];
        }
        device_in_adv_que_ = Omega_h::Write<Omega_h::I8>(host_in_adv_que_);
    }

    void PumiTallyImpl::search_initial_elements() {// assign the location to ptcl dest
        auto particle_dest = pumipic_ptcls->get<1>();
        auto in_flight     = pumipic_ptcls->get<3>();

        int64_t pumi_ps_size_l = pumi_ps_size;
        const auto& device_pos_buffer_l = device_pos_buffer_;

        auto set_particle_dest = PS_LAMBDA(const int &e, const int &pid, const int &mask){
            if (mask>0 && pid < pumi_ps_size_l){
                particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
                particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
                particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];

                // everyone is in flight for this initial search
                in_flight(pid) = 1;
            }
        };
        pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest, "set initial position as dest");

        // *initial* build and search to find the initial elements of the particles
        search_and_rebuild(true);
        is_pumipic_initialized = true;
    }

    [[maybe_unused]] [[deprecated]]
    void PumiTallyImpl::copy_to_device_position_buffer(const double *init_particle_positions) {
        for (int64_t pid = 0; pid < pumi_ps_size; pid++) {
            host_pos_buffer_[pid * 3 + 0] = init_particle_positions[pid * 3 + 0];
            host_pos_buffer_[pid * 3 + 1] = init_particle_positions[pid * 3 + 1];
            host_pos_buffer_[pid * 3 + 2] = init_particle_positions[pid * 3 + 2];
        }

        // copy to device buffer
        device_pos_buffer_ = Omega_h::Write<Omega_h::Real>(host_pos_buffer_);
    }

    void PumiTallyImpl::copy_data_to_device(double *init_particle_positions) {
        Kokkos::View<Omega_h::Real*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                host_pos_view(init_particle_positions, pumi_ps_size * 3);

        Kokkos::View<Omega_h::Real*, PPExeSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                device_pos_view(device_pos_buffer_.data(), pumi_ps_size * 3);

        Kokkos::deep_copy(device_pos_view, host_pos_view);
    }


    // methods for PumiTallyImpl and PumiParticleAtElemBoundary

    void PumiTallyImpl::init_pumi_libs(int& argc, char**& argv)
    {
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
                                  Omega_h::Write<Omega_h::LO> &xFace) {

        // TODO: make this a member variable of the struct
        const auto &side_is_exposed = Omega_h::mark_exposed_sides(&mesh);

        auto checkExposedEdges =
                PS_LAMBDA(const int e, const int pid, const int mask) {
                    if (mask > 0 && !ptcl_done[pid]) {
                        assert(lastExit[pid] != -1);
                        const Omega_h::LO bridge = lastExit[pid];
                        const bool exposed = side_is_exposed[bridge];
                        ptcl_done[pid] = exposed;
                        xFace[pid] = lastExit[pid];
                        //elem_ids[pid] = exposed ? -1 : elem_ids[pid];
                    }
                };
        pumipic::parallel_for(ptcls, checkExposedEdges, "apply vacumm boundary condition");
    }

    PumiParticleAtElemBoundary::PumiParticleAtElemBoundary(size_t nelems, size_t capacity)
            : flux_(nelems, 0.0, "flux"),
              prev_xpoint_(capacity * 3, 0.0, "prev_xpoint"), initial_(true) {
        printf(
                "[INFO] Particle handler at boundary with %d elements and %d "
                "x points size (3 * n_particles)\n",
                flux_.size(), prev_xpoint_.size());
    }

    void PumiParticleAtElemBoundary::operator()(Omega_h::Mesh &mesh, pumiinopenmc::PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
                                                Omega_h::Write<Omega_h::LO> &inter_faces, Omega_h::Write<Omega_h::LO> &lastExit,
                                                Omega_h::Write<Omega_h::Real> &inter_points, Omega_h::Write<Omega_h::LO> &ptcl_done) {
        apply_boundary_condition(mesh, ptcls, elem_ids, ptcl_done, lastExit, inter_faces);
        pp_move_to_new_element(mesh, ptcls, elem_ids, ptcl_done, lastExit);
        if (!initial_) {
            evaluateFlux(ptcls, inter_points);
        }
    }

    void PumiParticleAtElemBoundary::mark_initial_as(bool initial)
    {
        initial_ = initial;
    }

    void PumiParticleAtElemBoundary::updatePrevXPoint(Omega_h::Write<Omega_h::Real> &xpoints) {
        OMEGA_H_CHECK_PRINTF(
                xpoints.size() <= prev_xpoint_.size() && prev_xpoint_.size() != 0,
                "xpoints size %d is greater than prev_xpoint size %d\n", xpoints.size(),
                prev_xpoint_.size());
        auto prev_xpoint = prev_xpoint_;
        auto update = OMEGA_H_LAMBDA(Omega_h::LO i) { prev_xpoint[i] = xpoints[i]; };
        Omega_h::parallel_for(xpoints.size(), update, "update previous xpoints");
    }

    void PumiParticleAtElemBoundary::evaluateFlux(PPPS *ptcls, Omega_h::Write<Omega_h::Real> &xpoints) {
        //Omega_h::Real total_particles = ptcls->nPtcls();
        auto prev_xpoint = prev_xpoint_;
        auto flux = flux_;
        auto in_flight = ptcls->get<3>();

        auto evaluate_flux =
                PS_LAMBDA(const int &e, const int &pid, const int &mask) {
                    if (mask > 0) {
                        Omega_h::Vector<3> dest = {xpoints[pid * 3], xpoints[pid * 3 + 1],
                                                   xpoints[pid * 3 + 2]};
                        Omega_h::Vector<3> orig = {prev_xpoint[pid * 3], prev_xpoint[pid * 3 + 1],
                                                   prev_xpoint[pid * 3 + 2]};

                        Omega_h::Real parsed_dist = Omega_h::norm(dest - orig);  // / total_particles;
                        Kokkos::atomic_add(&flux[e], parsed_dist * in_flight(pid));
                    }
                };
        pumipic::parallel_for(ptcls, evaluate_flux, "flux evaluation loop");
    }

    Omega_h::Reals PumiParticleAtElemBoundary::normalizeFlux(Omega_h::Mesh &mesh) {
        const Omega_h::LO nelems = mesh.nelems();
        const auto &el2n = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;
        const auto &coords = mesh.coords();

        auto flux = flux_;

        Omega_h::Write<Omega_h::Real> tet_volumes(flux_.size(), -1.0, "tet_volumes");
        Omega_h::Write<Omega_h::Real> normalized_flux(flux_.size(), -1.0, "normalized flux");

        auto normalize_flux_with_volume = OMEGA_H_LAMBDA(Omega_h::LO elem_id) {
            const auto elem_verts = Omega_h::gather_verts<4>(el2n, elem_id);
            const auto elem_vert_coords = Omega_h::gather_vectors<4, 3>(coords, elem_verts);

            auto b = Omega_h::simplex_basis<3, 3>(elem_vert_coords);
            auto volume = Omega_h::simplex_size_from_basis(b);

            tet_volumes[elem_id] = volume;
            normalized_flux[elem_id] = flux[elem_id] / volume;
        };
        Omega_h::parallel_for(tet_volumes.size(), normalize_flux_with_volume,
                              "normalize flux");

        mesh.add_tag(Omega_h::REGION, "volume", 1, Omega_h::Reals(tet_volumes));
        return Omega_h::Reals(normalized_flux);
    }

    void PumiParticleAtElemBoundary::finalizeAndWritePumiFlux(Omega_h::Mesh& full_mesh, const std::string& filename){
        const auto& normalized_flux = normalizeFlux(full_mesh);
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

    void pumiRebuild(pumipic::Mesh* picparts, PPPS *ptcls, Omega_h::Write<Omega_h::LO>& elem_ids) {
        pumiUpdatePtclPositions(ptcls);
        pumipic::migrate_lb_ptcls(*picparts, ptcls, elem_ids, 1.05);
        pumipic::printPtclImb(ptcls);
    }

    // search and update parent elements
    //! @param initial initial search finds the initial location of the particles and doesn't tally
    void PumiTallyImpl::search_and_rebuild(bool initial){
        // initial cannot be false when is_pumipic_initialized is false
        // may fail if simulated more than one batch
        assert((is_pumipic_initialized == false && initial == true) || (is_pumipic_initialized == true && initial == false));
        p_pumi_particle_at_elem_boundary_handler->mark_initial_as(initial);
        Omega_h::LO maxLoops = 1000;
        auto orig = pumipic_ptcls->get<0>();
        auto dest = pumipic_ptcls->get<1>();
        auto pid  = pumipic_ptcls->get<2>();

        if (p_picparts_->mesh() == nullptr || p_picparts_->mesh()->nelems() == 0){
            printf("ERROR: Mesh is empty\n");
        }

        bool isFoundAll = pumipic::particle_search(*p_picparts_->mesh(), pumipic_ptcls.get(),
                                                   orig, dest, pid, elem_ids_, inter_faces_,
                                                   inter_points_, maxLoops, *p_pumi_particle_at_elem_boundary_handler);
        if (!isFoundAll){
            printf("ERROR: Not all particles are found. May need more loops in search\n");
        }
        if (!initial) {
            p_pumi_particle_at_elem_boundary_handler->updatePrevXPoint(inter_points_);
        }
        pumiRebuild(p_picparts_.get(), pumipic_ptcls.get(), elem_ids_);
    }


    std::unique_ptr<PPPS> pp_create_particle_structure(Omega_h::Mesh mesh, pumipic::lid_t numPtcls){
        Omega_h::Int ne = mesh.nelems();
        pumiinopenmc::PPPS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
        pumiinopenmc::PPPS::kkGidView element_gids("element_gids", ne);

        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy;

        Omega_h::parallel_for(
                ne, OMEGA_H_LAMBDA(const Omega_h::LO &i) { element_gids(i) = i; });

        Omega_h::parallel_for(mesh.nelems(),
                              OMEGA_H_LAMBDA(Omega_h::LO id){
                                  ptcls_per_elem[id] = (id==0) ? numPtcls : 0;
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

    void start_pumi_particles_in_0th_element(Omega_h::Mesh& mesh, pumiinopenmc::PPPS* ptcls) {
        // find the centroid of the 0th element
        const auto& coords = mesh.coords();
        const auto& tet2node = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

        Omega_h::Write<Omega_h::Real> centroid_of_el0(3, 0.0, "centroid");

        auto find_centroid_of_el0 = OMEGA_H_LAMBDA(Omega_h::LO id) {
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
        auto pids     = ptcls->get<2>();
        auto in_fly   = ptcls->get<3>();

        auto set_initial_positions = PS_LAMBDA(const int &e, const int &pid, const int &mask) {
            if (mask>0) {
                pids(pid)          = pid;
                in_fly(pid)        = 1;
                init_loc(pid, 0) = centroid_of_el0[0];
                init_loc(pid, 1) = centroid_of_el0[1];
                init_loc(pid, 2) = centroid_of_el0[2];
            }
        };
        pumipic::parallel_for(ptcls, set_initial_positions, "set initial particle positions");
    }

    Omega_h::Mesh* PumiTallyImpl::partition_pumipic_mesh()
    {
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

    void PumiTallyImpl::create_and_initialize_pumi_particle_structure(Omega_h::Mesh* mesh)
    {
        pumipic_ptcls = pp_create_particle_structure(*mesh, pumi_ps_size);
        start_pumi_particles_in_0th_element(*mesh, pumipic_ptcls.get());
        p_pumi_particle_at_elem_boundary_handler =
                std::make_unique<pumiinopenmc::PumiParticleAtElemBoundary>(mesh->nelems(),
                                                                           pumipic_ptcls->capacity());

        printf("PumiPIC Mesh and data structure created with %d and %d as particle structure capacity\n",
               p_picparts_->mesh()->nelems(), pumipic_ptcls->capacity());
    }

    void PumiTallyImpl::read_pumipic_lib_and_full_mesh(int& argc, char**& argv)
    {
        printf("Reading the Omega_h mesh %s to tally with tracklength estimator\n", oh_mesh_fname.c_str());
        init_pumi_libs(argc, argv);

        if (oh_mesh_fname.empty()){
            printf("[ERROR] Omega_h mesh for PumiPIC is not given. Provide --ohMesh = <osh file>");
        }
        full_mesh_ = Omega_h::binary::read(oh_mesh_fname, &oh_lib);
        if (full_mesh_.dim() != 3){
            printf("PumiPIC only works for 3D mesh now.\n");
        }
        printf("PumiPIC Loaded mesh %s with %d elements\n", oh_mesh_fname.c_str(), full_mesh_.nelems());
    }

    void PumiTallyImpl::load_pumipic_mesh_and_init_particles(int& argc, char**& argv) {
        read_pumipic_lib_and_full_mesh(argc, argv);
        Omega_h::Mesh* mesh = partition_pumipic_mesh();
        create_and_initialize_pumi_particle_structure(mesh);
    }

    PumiTally::PumiTally(std::string& mesh_filename, int64_t num_particles, int& argc, char**& argv)
            : pimpl(std::make_unique<PumiTallyImpl>(mesh_filename, num_particles, argc, argv)) {
    }

    void PumiTally::initialize_particle_location(double* init_particle_positions, int64_t size){
        auto start_time = std::chrono::steady_clock::now();

        pimpl->initialize_particle_location(init_particle_positions, size);

        std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start_time;
        pimpl->tally_times.initialization_time += elapsed_seconds.count();
    }

    void PumiTally::move_to_next_location(double* particle_destinations, int8_t* flying, int64_t size){
        auto start_time = std::chrono::steady_clock::now();

        pimpl->move_to_next_location(particle_destinations, flying, size);

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