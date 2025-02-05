//
// Created by Fuad Hasan on 2/3/25.
//
#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_vtk.hpp>

// TODO: Remove it by having another header file
#include "pumipic_particle_data_structure.cpp"

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

    // create particle structure with 5 particles
    std::unique_ptr<pumiinopenmc::PumiTallyImpl> p_pumi_tallyimpl = std::make_unique<pumiinopenmc::PumiTallyImpl>(temp_file_name, num_ptcls,
                                                                                                      argc, argv);
    fprintf(stdout, "[INFO] Particle structure created successfully\n");

    SECTION("Check if elem_ids_ member is set properly..."){
        auto elem_ids_l = p_pumi_tallyimpl->elem_ids_;
        REQUIRE(elem_ids_l.size() == p_pumi_tallyimpl->pumipic_ptcls->nPtcls());
        REQUIRE(elem_ids_l.size() <= p_pumi_tallyimpl->pumipic_ptcls->capacity());
    }
}