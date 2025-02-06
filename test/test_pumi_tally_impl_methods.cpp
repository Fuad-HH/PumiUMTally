//
// Created by Fuad Hasan on 2/3/25.
//
#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_vtk.hpp>

// ********************************** Notes ************************************************************//
// - Sections are not working for some reason. Saying using MPI functions before or after MPI init or finalize
// - sections are marked with comments for now
// - TODO Remove including cpp by creating another internal header
// - Look at this gist to verify this in python: https://gist.github.com/Fuad-HH/5e0aed99f271617e283e9108091fb1cb
// *****************************************************************************************************//

// TODO: Remove it by having another header file
#include "pumipic_particle_data_structure.cpp"

bool is_close(const double a, const double b, double tol = 1e-8) {
    return std::abs(a - b) < tol;
}

OMEGA_H_INLINE bool is_close_d(const double a, const double b, double tol = 1e-8) {
    return Kokkos::abs(a - b) < tol;
}

TEST_CASE("Test Impl Class Functions") {
    auto lib = Omega_h::Library{};
    auto world = lib.world();
    // simplest 3D mesh
    auto mesh =
            Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 1, 1, 1, false);
    printf("[INFO] Mesh created with %d vertices and %d faces\n",
           mesh.nverts(), mesh.nfaces());

    // create particle structure with 5 particles
    int num_ptcls = 5;

    // TODO remove this read_write
    int argc = 0;
    char **argv;
    std::string temp_file_name = "mesh.osh";
    Omega_h::binary::write(temp_file_name, &mesh);
    fprintf(stdout, "[INFO] Mesh written to file :%s\n", temp_file_name.c_str());

    //****************************** Checks Regarding Constructor ******************************************//
    //*******************************************************************************************************//

    // create particle structure with 5 particles
    std::unique_ptr<pumiinopenmc::PumiTallyImpl> p_pumi_tallyimpl = std::make_unique<pumiinopenmc::PumiTallyImpl>(
            temp_file_name, num_ptcls,
            argc, argv);
    fprintf(stdout, "[INFO] Particle structure created successfully\n");

    {// * Check element IDs
        auto elem_ids_l = p_pumi_tallyimpl->elem_ids_;
        REQUIRE(elem_ids_l.size() == p_pumi_tallyimpl->pumipic_ptcls->nPtcls());
        REQUIRE(elem_ids_l.size() <= p_pumi_tallyimpl->pumipic_ptcls->capacity());
        Omega_h::HostWrite<Omega_h::LO> elem_ids_host(elem_ids_l);
        for (int i = 0; i < elem_ids_host.size(); ++i) {
            REQUIRE(elem_ids_host[i] == 0);
        }
    }

    {// * Check other array sizes
        auto device_pos_l = p_pumi_tallyimpl->device_pos_buffer_;
        auto device_adv_l = p_pumi_tallyimpl->device_in_adv_que_;
        auto device_wgt_l = p_pumi_tallyimpl->weights_;

        REQUIRE(device_pos_l.size() == num_ptcls * 3);
        REQUIRE(device_adv_l.size() == num_ptcls);
        REQUIRE(device_wgt_l.size() == num_ptcls);

        // * Check full mesh
        REQUIRE(p_pumi_tallyimpl->full_mesh_.nelems() == 6);
        REQUIRE(p_pumi_tallyimpl->full_mesh_.dim() == 3);

        // * Check the picparts
        REQUIRE(p_pumi_tallyimpl->p_picparts_->isFullMesh() == true);

        // * Check created particle structure
        REQUIRE(p_pumi_tallyimpl->pumipic_ptcls->nPtcls() == num_ptcls);
        REQUIRE(p_pumi_tallyimpl->pumipic_ptcls->capacity() >= num_ptcls);
        REQUIRE(p_pumi_tallyimpl->pumipic_ptcls->nElems() == mesh.nelems());
    }

    {// * Check if particles have origin at 0th element
        auto origin = p_pumi_tallyimpl->pumipic_ptcls->get<0>();
        Omega_h::Vector<3> cell_centroid_of_elem0{0.5, 0.75, 0.25};
        Omega_h::Write<Omega_h::Real> cell_centroids(p_pumi_tallyimpl->pumipic_ptcls->nPtcls() * 3, 0.0, "check_sum");
        auto check_init_at_0th_elem = PS_LAMBDA(const auto &el, const auto &pid, const auto &mask) {
            if (mask > 0) {
                cell_centroids[pid * 3] = origin(pid, 0);
                cell_centroids[pid * 3 + 1] = origin(pid, 1);
                cell_centroids[pid * 3 + 2] = origin(pid, 2);
            }
        };
        pumipic::parallel_for(p_pumi_tallyimpl->pumipic_ptcls.get(), check_init_at_0th_elem,
                              "check if particles are intialized at 0th element");
        Omega_h::HostWrite<Omega_h::Real> cell_centroids_host(cell_centroids);
        for (int pid = 0; pid < num_ptcls; ++pid) {
            printf("Particle %d position: (%.16f, %.16f, %.16f)\n", pid,
                   cell_centroids_host[pid * 3],
                   cell_centroids_host[pid * 3 + 1],
                   cell_centroids_host[pid * 3 + 2]);
            REQUIRE(is_close(cell_centroids_host[pid * 3], cell_centroid_of_elem0[0]));
            REQUIRE(is_close(cell_centroids_host[pid * 3 + 1], cell_centroid_of_elem0[1]));
            REQUIRE(is_close(cell_centroids_host[pid * 3 + 2], cell_centroid_of_elem0[2]));
        }
    }

    //************************************ Checks Regarding Initializing Particle Locations ***********************//
    //*************************************************************************************************************//

    {// * Note: all 5 particles will start their journey as follows:
        // ray_origin   =   [0.1, 0.4, 0.5] in cell 2
        // ray_end      =   [1.1, 0.4, 0.5] passing through cells 2, 3, 4 and finally leaving the box
        std::vector<double> init_particle_positions(num_ptcls * 3);
        for (int pid = 0; pid < num_ptcls; ++pid) {
            init_particle_positions[pid * 3] = 0.1;
            init_particle_positions[pid * 3 + 1] = 0.4;
            init_particle_positions[pid * 3 + 2] = 0.5;
        }
        p_pumi_tallyimpl->initialize_particle_location(init_particle_positions.data(), init_particle_positions.size());

        {// * Check if particle positions are copied properly in the device
            // ? is it okay to check with OMEGA_H_CHECK
            auto device_pos_buffer_l = p_pumi_tallyimpl->device_pos_buffer_;
            auto check_device_init_pos = OMEGA_H_LAMBDA(int pid) {
                OMEGA_H_CHECK_PRINTF(is_close_d(device_pos_buffer_l[pid * 3], 0.1),
                                     "Particle position copy to device error 0: %.16f %.16f\n",
                                     device_pos_buffer_l[pid * 3], 0.1);
                OMEGA_H_CHECK_PRINTF(is_close_d(device_pos_buffer_l[pid * 3 + 1], 0.4),
                                     "Particle position copy to device error 0: %.16f %.16f\n",
                                     device_pos_buffer_l[pid * 3 + 1], 0.4);
                OMEGA_H_CHECK_PRINTF(is_close_d(device_pos_buffer_l[pid * 3 + 2], 0.5),
                                     "Particle position copy to device error 0: %.16f %.16f\n",
                                     device_pos_buffer_l[pid * 3 + 2], 0.5);
            };
            Omega_h::parallel_for(num_ptcls, check_device_init_pos,
                                  "Check if the init particle pos are copied to device correctly");
        }
    }

    {// * Check if all particles reached element 2
        auto elem_ids_host = Omega_h::HostWrite<Omega_h::LO>(p_pumi_tallyimpl->elem_ids_);
        for (int pid = 0; pid < num_ptcls; ++pid) {
            REQUIRE(elem_ids_host[pid] == 2);
        }

        // * The fluxes should be zero since the init doesn't calculate flux
        auto flux_l = p_pumi_tallyimpl->p_pumi_particle_at_elem_boundary_handler->flux_;
        Omega_h::HostWrite<Omega_h::Real> flux_host(flux_l);
        REQUIRE(flux_host.size() == mesh.nelems());
        for (int el = 0; el < flux_host.size(); ++el) {
            printf("Flux after init run %d[%.16f]\n", el, flux_host[el]);
            REQUIRE(is_close(flux_host[el], 0.0));
        }
    }

    //******************************* Checks Move to Next Location *************************************************//
    //**************************************************************************************************************//

    // All the particles will now go to ray end
    std::vector<double> particle_destination(num_ptcls * 3);
    std::vector<double> weights(num_ptcls, 1.0); // same weights

    for (int pid = 0; pid < num_ptcls; ++pid) {
        particle_destination[pid * 3] = 1.1;
        particle_destination[pid * 3 + 1] = 0.4;
        particle_destination[pid * 3 + 2] = 0.5;
    }


    { // * Check copy data to device and reset flying
        std::vector<int8_t> flying(num_ptcls, 1); // all are flying now

        REQUIRE(particle_destination.size() == 3 * p_pumi_tallyimpl->pumi_ps_size);
        p_pumi_tallyimpl->copy_data_to_device(particle_destination.data());
        p_pumi_tallyimpl->copy_and_reset_flying_flag(flying.data());
        p_pumi_tallyimpl->copy_weights(weights.data());

        auto particle_destinations_l = p_pumi_tallyimpl->device_pos_buffer_;
        Omega_h::HostWrite<Omega_h::Real> particle_destination_l_host(particle_destinations_l);
        auto particle_weight_l = p_pumi_tallyimpl->weights_;
        Omega_h::HostWrite<Omega_h::Real> particle_weight_l_host(particle_weight_l);
        auto particle_flying_l = p_pumi_tallyimpl->device_in_adv_que_;
        Omega_h::HostWrite<Omega_h::I8> particle_flying_l_host(particle_flying_l);

        for (int pid = 0; pid < num_ptcls; ++pid) {
            REQUIRE(is_close(particle_destination_l_host[pid * 3], 1.1));
            REQUIRE(is_close(particle_destination_l_host[pid * 3 + 1], 0.4));
            REQUIRE(is_close(particle_destination_l_host[pid * 3 + 2], 0.5));

            REQUIRE(is_close(particle_weight_l_host[pid], 1.0));
            REQUIRE(particle_flying_l_host[pid] == 1);

            REQUIRE(flying[pid] == 0);
        }
    }

    std::vector<int8_t> flying(num_ptcls, 1); // reset them again to 1
    p_pumi_tallyimpl->move_to_next_location(particle_destination.data(), flying.data(), weights.data(),
                                            particle_destination.size());

    {// * Check if the particles correctly reaches element 4
        auto elem_ids_local = p_pumi_tallyimpl->elem_ids_;
        Omega_h::HostWrite<Omega_h::LO> elem_ids_local_host(elem_ids_local);
        for (int pid = 0; pid < num_ptcls; ++pid) {
            printf("[INFO] Particles reached elem %d\n", elem_ids_local_host[pid]);
            REQUIRE(elem_ids_local_host[pid] == 4);
        }
    }

    {// * Check if particles reached destinations properly, weights and flying flags are copied properly
        auto new_origin = p_pumi_tallyimpl->pumipic_ptcls->get<0>();
        auto particle_flyign = p_pumi_tallyimpl->pumipic_ptcls->get<3>();
        auto particle_weights = p_pumi_tallyimpl->pumipic_ptcls->get<4>();
        auto check_copied_properties = PS_LAMBDA(const auto &e, const auto &pid, const auto &mask) {
            if (mask > 0) {
                printf("Particle new origin (%f, %f, %f), flying %d, weight %f\n", new_origin(pid, 0),
                       new_origin(pid, 1), new_origin(pid, 2),
                       particle_flyign(pid), particle_weights(pid));
                // fixme The new position should be 1.0 rather than 1.1 since it goes out
                OMEGA_H_CHECK_PRINTF(is_close_d(new_origin(pid, 0), 1.1),
                                     "Particle destination not copied properly %.16f\n", new_origin(pid, 0));
                OMEGA_H_CHECK_PRINTF(is_close_d(new_origin(pid, 1), 0.4),
                                     "Particle destination not copied properly %.16f\n", new_origin(pid, 1));
                OMEGA_H_CHECK_PRINTF(is_close_d(new_origin(pid, 2), 0.5),
                                     "Particle destination not copied properly %.16f\n", new_origin(pid, 2));

                OMEGA_H_CHECK_PRINTF(particle_flyign(pid) == 1, "Particle flying not copied correctly, found %d\n",
                                     particle_flyign(pid));
                OMEGA_H_CHECK_PRINTF(is_close_d(particle_weights(pid), 1.0),
                                     "Particle weight not copied properly, found %.16f\n", particle_weights(pid));
            }
        };
        pumipic::parallel_for(p_pumi_tallyimpl->pumipic_ptcls.get(), check_copied_properties,
                              "check if data copied before move");
    }

    {// * Check flux
        // * Note: The particles are going through elems 2, 3, 4. The lengths are:
        // 0.3, 0.1, and 0.5 (times 5 for 5 particles)
        auto flux_local = p_pumi_tallyimpl->p_pumi_particle_at_elem_boundary_handler->flux_;
        Omega_h::HostWrite<Omega_h::Real> flux_host(flux_local);
        printf("The fluxes are %d[%f] %d[%f] %d[%f] %d[%f] %d[%f] %d[%f]", 0, flux_host[0], 1, flux_host[1], 2,
               flux_host[2], 3, flux_host[3], 4, flux_host[4], 5, flux_host[5]);
        REQUIRE(is_close(flux_host[0], 0.0));
        REQUIRE(is_close(flux_host[1], 0.0));
        REQUIRE(is_close(flux_host[2], 0.3 * num_ptcls));
        REQUIRE(is_close(flux_host[3], 0.1 * num_ptcls));
        REQUIRE(is_close(flux_host[4], 0.5 * num_ptcls));
        REQUIRE(is_close(flux_host[5], 0.0));
    }
}

TEST_CASE("Test Boundary Handler Struct and Operator") {
    // ********************************** Set UP *********************************************************************//
    auto lib = Omega_h::Library{};
    auto world = lib.world();
    // simplest 3D mesh
    auto mesh =
            Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 1, 1, 1, false);
    printf("[INFO] Mesh created with %d vertices and %d faces\n",
           mesh.nverts(), mesh.nfaces());

    // create particle structure with 5 particles
    int num_ptcls = 5;

    // TODO remove this read_write
    int argc = 0;
    char **argv;
    std::string temp_file_name = "mesh.osh";
    Omega_h::binary::write(temp_file_name, &mesh);
    fprintf(stdout, "[INFO] Mesh written to file :%s\n", temp_file_name.c_str());


    // create particle structure with 5 particles
    std::unique_ptr<pumiinopenmc::PumiTallyImpl> p_pumi_tallyimpl = std::make_unique<pumiinopenmc::PumiTallyImpl>(
            temp_file_name, num_ptcls,
            argc, argv);
    fprintf(stdout, "[INFO] Particle structure created successfully\n");

    std::vector<double> init_particle_positions(num_ptcls * 3);
    for (int pid = 0; pid < num_ptcls; ++pid) {
        init_particle_positions[pid * 3] = 0.1;
        init_particle_positions[pid * 3 + 1] = 0.4;
        init_particle_positions[pid * 3 + 2] = 0.5;
    }

    // this particle structure will be used to check the operator()
    p_pumi_tallyimpl->initialize_particle_location(init_particle_positions.data(), init_particle_positions.size());

    {
        {// *Check if elem_ids_ are 2 now
            auto elem_ids_l = p_pumi_tallyimpl->elem_ids_;
            Omega_h::HostWrite<Omega_h::LO> elem_id_h(elem_ids_l);
            for (int pid = 0; pid < num_ptcls; ++pid) {
                REQUIRE(elem_id_h[pid] == 2);
            }
        }

        // initialize inter-faces

        // initialize lastExit

        {// test prev_xpoint
            auto lastExit_l = p_pumi_tallyimpl->p_pumi_particle_at_elem_boundary_handler->prev_xpoint_;
            Omega_h::HostWrite<Omega_h::Real> lastExit_host(lastExit_l);
            REQUIRE(lastExit_host.size() == 3 * p_pumi_tallyimpl->pumipic_ptcls->capacity());
            // prev_xpoints should be the current positions
        }

        // initialize inter_points: should be the initial positions

        // initialize ptcl_done
        Omega_h::Write<Omega_h::LO> ptcl_done(num_ptcls, 0, "particle done flag");

        {// * check how inter_points look like
            auto inter_points_l = p_pumi_tallyimpl->inter_points_;
            Omega_h::HostWrite<Omega_h::Real> inter_points_host(inter_points_l);
            REQUIRE(inter_points_host.size() == 0); // uninitialized
        }
    }


}
