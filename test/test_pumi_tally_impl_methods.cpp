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
// *****************************************************************************************************//

// TODO: Remove it by having another header file
#include "pumipic_particle_data_structure.cpp"

bool is_close(const double a, const double b, double tol = 1e-8){
    return std::abs(a-b) < tol;
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

    // * Check element IDs
    auto elem_ids_l = p_pumi_tallyimpl->elem_ids_;
    REQUIRE(elem_ids_l.size() == p_pumi_tallyimpl->pumipic_ptcls->nPtcls());
    REQUIRE(elem_ids_l.size() <= p_pumi_tallyimpl->pumipic_ptcls->capacity());
    Omega_h::HostWrite<Omega_h::LO> elem_ids_host(elem_ids_l);
    for (int i = 0; i < elem_ids_host.size(); ++i) {
        REQUIRE(elem_ids_host[i] == 0);
    }

    // * Check other array sizes
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

    // * Check if particles have origin at 0th element
    auto origin = p_pumi_tallyimpl->pumipic_ptcls->get<0>();
    Omega_h::Vector<3> cell_centroid_of_elem0 {0.5,  0.75, 0.25};
    Omega_h::Write<Omega_h::Real> cell_centroids(p_pumi_tallyimpl->pumipic_ptcls->nPtcls()*3, 0.0, "check_sum");
    auto check_init_at_0th_elem = PS_LAMBDA(const auto& el, const auto& pid, const auto& mask){
        if (mask > 0) {
            cell_centroids[pid * 3] = origin(pid, 0);
            cell_centroids[pid * 3 + 1] = origin(pid, 1);
            cell_centroids[pid * 3 + 2] = origin(pid, 2);
        }
    };
    pumipic::parallel_for(p_pumi_tallyimpl->pumipic_ptcls.get(), check_init_at_0th_elem,
                          "check if particles are intialized at 0th element");
    Omega_h::HostWrite<Omega_h::Real> cell_centroids_host(cell_centroids);
    for(int pid = 0; pid<num_ptcls; ++pid){
        printf("Particle %d position: (%.16f, %.16f, %.16f)\n", pid,
               cell_centroids_host[pid*3],
               cell_centroids_host[pid*3+1],
               cell_centroids_host[pid*3+2]);
        REQUIRE(is_close(cell_centroids_host[pid*3], cell_centroid_of_elem0[0]));
        REQUIRE(is_close(cell_centroids_host[pid*3+1], cell_centroid_of_elem0[1]));
        REQUIRE(is_close(cell_centroids_host[pid*3+2], cell_centroid_of_elem0[2]));
    }

    //************************************ Checks Regarding Initializing Particle Locations ***********************//
    //*************************************************************************************************************//
}