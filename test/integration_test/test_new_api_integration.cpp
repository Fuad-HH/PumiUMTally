//
// Created by Fuad Hasan on 6/19/26.
//
// Integration test full PUMI-Tally API.
//
// Mesh: test/assets/tet6-222.osh  (6-tet Omega_h mesh, 3D)
// Particles: 4, each starting at a distinct non-random location.
//

#include <Omega_h_library.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include <PumiTallyImpl.h>

// ---------------------------------------------------------------------------
// Helper: floating-point approximate comparison (host)
// ---------------------------------------------------------------------------
bool is_close(const double a, const double b, const double tol = 1e-8) {
  return std::abs(a - b) < tol;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int kNumPtcls = 4;
constexpr int kNelemsExpected = 6; // tet6 mesh
constexpr int kMeshDim = 3;

#ifdef TEST_MESH_FILE
const std::string kMeshFilename = TEST_MESH_FILE;
#else
const std::string kMeshFilename = "test/assets/tet6-222.osh";
#endif

// clang-format off
//
//  Starting positions (hand-picked, non-random, well inside the [0,2]³ mesh):
//    P0 : (0.60, 0.60, 0.60)   — near origin corner
//    P1 : (1.40, 0.60, 0.60)   — offset in +x
//    P2 : (0.60, 1.40, 0.60)   — offset in +y
//    P3 : (0.60, 0.60, 1.40)   — offset in +z
//
//  Displacement per move: (+0.10, +0.10, +0.10)
//
//  Move 1:  P0->(0.70,0.70,0.70)  P1->(1.50,0.70,0.70)
//           P2->(0.70,1.50,0.70)  P3->(0.70,0.70,1.50)
//  Move 2:  P0->(0.80,0.80,0.80)  P1->(1.60,0.80,0.80)
//           P2->(0.80,1.60,0.80)  P3->(0.80,0.80,1.60)
//  Move 3:  P0->(0.90,0.90,0.90)  P1->(1.70,0.90,0.90)
//           P2->(0.90,1.70,0.90)  P3->(0.90,0.90,1.70)
//  Move 4 (reflective BC):
//           P0->(1.10,1.10,1.10) flying=1 (inside)
//           P1->(2.10,0.90,0.90) flying=1 (x>2, reflects)
//           P2->(0.90,1.70,0.90) flying=0 (stays)
//           P3->(0.90,1.30,2.10) flying=1 (z>2, reflects)
// clang-format on

// NOTE: non-const — API functions take double*, int8_t* (non-const pointers)
// clang-format off
std::vector<double> kInitPos = {
    0.60, 0.60, 0.60, // P0
    1.40, 0.60, 0.60, // P1
    0.60, 1.40, 0.60, // P2
    0.60, 0.60, 1.40  // P3
};
std::vector<double> kDest1 = {
    0.70, 0.70, 0.70, 1.50, 0.70, 0.70,
    0.70, 1.50, 0.70, 0.70, 0.70, 1.50
};
std::vector<double> kDest2 = {
    0.80, 0.80, 0.80, 1.60, 0.80, 0.80,
    0.80, 1.60, 0.80, 0.80, 0.80, 1.60
};
std::vector<double> kDest3 = {
    0.90, 0.90, 0.90, 1.70, 0.90, 0.90,
    0.90, 1.70, 0.90, 0.90, 0.90, 1.70
};
// clang-format on

// Helper: copy particle origins from particle DS to an Omega_h Write array
void copy_origins_to_array(pumitally::PPPS *ptcls,
                           Omega_h::Write<Omega_h::Real> &out) {
  auto ptcl_origin = ptcls->get<0>();
  auto copy_fn = PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0) {
      out[pid * 3 + 0] = ptcl_origin(pid, 0);
      out[pid * 3 + 1] = ptcl_origin(pid, 1);
      out[pid * 3 + 2] = ptcl_origin(pid, 2);
    }
  };
  pumipic::parallel_for(ptcls, copy_fn, "copy origins");
}

// Helper: verify all particles are in valid elements
void require_valid_elements(pumitally::PumiTallyImpl &tally,
                            const char *label) {
  auto elem_ids = tally.p_particle_tracer->getElementIds();
  auto host = Omega_h::HostRead<Omega_h::LO>(elem_ids);
  for (int pid = 0; pid < kNumPtcls; ++pid) {
    INFO(label << " — Particle " << pid << " element = " << host[pid]);
    REQUIRE(host[pid] >= 0);
    REQUIRE(host[pid] < tally.full_mesh.nelems());
  }
}

// Helper: verify particle positions match expected
void require_positions_match(pumitally::PPPS *ptcls,
                             const std::vector<double> &expected,
                             const char *label) {
  Omega_h::Write<Omega_h::Real> pos(kNumPtcls * 3, 0.0, "pos_check");
  copy_origins_to_array(ptcls, pos);
  auto host = Omega_h::HostRead<Omega_h::Real>(pos);
  for (int pid = 0; pid < kNumPtcls; ++pid) {
    INFO(label << " — Particle " << pid << " position ("
         << host[pid * 3 + 0] << ", " << host[pid * 3 + 1] << ", "
         << host[pid * 3 + 2] << ")  expected ("
         << expected[pid * 3 + 0] << ", " << expected[pid * 3 + 1] << ", "
         << expected[pid * 3 + 2] << ")");
    REQUIRE(is_close(host[pid * 3 + 0], expected[pid * 3 + 0]));
    REQUIRE(is_close(host[pid * 3 + 1], expected[pid * 3 + 1]));
    REQUIRE(is_close(host[pid * 3 + 2], expected[pid * 3 + 2]));
  }
}

// ===========================================================================
// Single integration test — all PUMI-Tally capabilities sequentially
// ===========================================================================
TEST_CASE("Test PUMI-Tally Multi-Dimensional Tally API", "[integration]") {

  auto lib = Omega_h::Library{};
  int argc = 0;
  char **argv = nullptr;

  // ================================================================
  // Step 1 — Construction & mesh verification
  // ================================================================
  printf("\n===== Step 1: Construct PumiTallyImpl and verify mesh =====\n");
  {
    auto pumi_tally = std::make_unique<pumitally::PumiTallyImpl>(
        kMeshFilename, kNumPtcls, argc, argv);

    REQUIRE(pumi_tally->full_mesh.nelems() == kNelemsExpected);
    REQUIRE(pumi_tally->full_mesh.dim() == kMeshDim);
    REQUIRE(pumi_tally->num_particles == kNumPtcls);
    REQUIRE(pumi_tally->pumipic_ptcls != nullptr);
    REQUIRE(pumi_tally->pumipic_ptcls->nPtcls() == kNumPtcls);
    REQUIRE(pumi_tally->pumipic_ptcls->capacity() >= kNumPtcls);
    REQUIRE(pumi_tally->pumipic_ptcls->nElems() == kNelemsExpected);
    REQUIRE(pumi_tally->position_dev_buffer.size() == kNumPtcls * 3);
    REQUIRE(pumi_tally->flying_dev_buffer.size() == kNumPtcls);
    REQUIRE(pumi_tally->weights_dev_buffer.size() == kNumPtcls);

    printf("[PASS] Step 1: Mesh and particle structure verified\n");
  }

  // ================================================================
  // Step 2 — Register multi-dimensional tallies
  // ================================================================
  printf("\n===== Step 2: Register element and node tallies =====\n");
  {
    auto pumi_tally = std::make_unique<pumitally::PumiTallyImpl>(
        kMeshFilename, kNumPtcls, argc, argv);

    pumi_tally->AddElementTally({2, 3}); // 2 energy × 3 angle = 6 bins
    pumi_tally->AddNodeTally({2, 3});    // same at nodes

    const auto &boundary_handler = pumi_tally->p_pumi_particle_at_elem_boundary_handler;
    REQUIRE(boundary_handler != nullptr);
    REQUIRE(boundary_handler->HasMultiDimTally() == true);
    REQUIRE(boundary_handler->element_tally_spec.is_initialized == true);
    REQUIRE(boundary_handler->element_tally_spec.bins_per_filter.size() == 2);
    REQUIRE(boundary_handler->element_tally_spec.total_filter_bins == 6);
    REQUIRE(boundary_handler->node_tally_spec.is_initialized == true);
    REQUIRE(boundary_handler->node_tally_spec.total_filter_bins == 6);

    const auto nelem = pumi_tally->full_mesh.nelems();
    const auto nvert = pumi_tally->full_mesh.nverts();
    REQUIRE(boundary_handler->element_tallies.size() == static_cast<size_t>(nelem) * 6);
    REQUIRE(boundary_handler->node_tallies.size() == static_cast<size_t>(nvert) * 6);

    printf("[PASS] Step 2: Element tally %d×6, Node tally %d×6 registered\n",
           nelem, nvert);
  }

  // ================================================================
  // Step 3 — Initial position search
  // ================================================================
  printf("\n===== Step 3: Initial position search =====\n");
  {
    auto pumi_tally = std::make_unique<pumitally::PumiTallyImpl>(
        kMeshFilename, kNumPtcls, argc, argv);

    pumi_tally->CopyInitialPositionToBuffer(
        kInitPos.data(), static_cast<Omega_h::LO>(kInitPos.size()));

    require_valid_elements(*pumi_tally, "Step 3");
    printf("[PASS] Step 3: All %d particles placed in valid elements\n",
           kNumPtcls);

    // Assert that the tally results are all zero
    const auto &boundary_handler = pumi_tally->p_pumi_particle_at_elem_boundary_handler;
    auto tallies_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                            boundary_handler->element_tallies);
    for (size_t i = 0; i < tallies_host.size(); ++i)
      REQUIRE_THAT(tallies_host(i), Catch::Matchers::WithinAbs(0.0, 1e-6));

    tallies_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                            boundary_handler->node_tallies);
    for (size_t i = 0; i < tallies_host.size(); ++i)
      REQUIRE_THAT(tallies_host(i), Catch::Matchers::WithinAbs(0.0, 1e-6));

    printf("[PASS] Step 3: Tally results are all zero\n");
  }

  // ================================================================
  // Step 4 — Single move, verify positions and flux
  //
  // Hand calc: segment = √(0.1²+0.1²+0.1²) ≈ 0.1732
  //            total flux ≈ 4 × 0.1732 ≈ 0.6928
  // ================================================================
  printf("\n===== Step 4: First move (+0.10,+0.10,+0.10) =====\n");
  {
    auto pumi_tally = std::make_unique<pumitally::PumiTallyImpl>(
        kMeshFilename, kNumPtcls, argc, argv);

    pumi_tally->AddElementTally({1});
    std::vector<unsigned int> bins(kNumPtcls, 0);
    pumi_tally->UpdateFilterBins(bins);

    pumi_tally->CopyInitialPositionToBuffer(
        kInitPos.data(), static_cast<Omega_h::LO>(kInitPos.size()));

    std::vector<int8_t> flying(kNumPtcls, 1);
    std::vector<double> weights(kNumPtcls, 1.0);

    pumi_tally->MoveToNextLocation(
        kInitPos.data(), kDest1.data(), flying.data(), weights.data(),
        static_cast<int64_t>(kDest1.size()));

    require_valid_elements(*pumi_tally, "Step 4");
    require_positions_match(pumi_tally->pumipic_ptcls.get(), kDest1, "Step 4");

    // Verify flux > 0
    const auto &h = pumi_tally->p_pumi_particle_at_elem_boundary_handler;
    auto tallies_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                            h->element_tallies);
    double total_flux = 0.0;
    for (size_t i = 0; i < tallies_host.size(); ++i)
      total_flux += tallies_host(i);
    printf("[INFO] Step 4 total flux: %.6f (expected ~0.6928)\n", total_flux);
    REQUIRE_THAT(total_flux, Catch::Matchers::WithinAbs(4 * std::sqrt(3*0.1*0.1), 1e-6));
    printf("[PASS] Step 4: Move 1 complete, positions verified, flux "
           "accumulated\n");
  }

  // ================================================================
  // Step 5 — Two sequential moves with multi-dimensional tally
  //
  // Filter bins: P0,P1->(0,0), P2,P3->(1,2)
  // Total flux after 2 moves = 2 x 4 x sqrt(3*0.1*0.1) ≈ 1.3856
  // In Tallies: (all, 0, 0) = 1 x 4 x sqrt(3*0.1*0.1) ≈ 0.6928
  //             (all, 1, 2) = 1 x 4 x sqrt(3*0.1*0.1) ≈ 0.6928
  // ================================================================

  printf("\n===== Step 5: Two sequential moves + multi-dim tally =====\n");
  {
    const double expected_total_flux = 2 * 4 * std::sqrt(3*0.1*0.1);
    const double expected_flux_all_0_0 = expected_total_flux / 2;
    const double expected_flux_all_1_2 = expected_total_flux / 2;

    auto pumi_tally = std::make_unique<pumitally::PumiTallyImpl>(
        kMeshFilename, kNumPtcls, argc, argv);

    pumi_tally->AddElementTally({2, 3});

    std::vector<unsigned int> bins(kNumPtcls * 2);
    bins[0] = 0; bins[1] = 0; // P0 in energy bin 0, angle bin 0
    bins[2] = 0; bins[3] = 0; // P1 in energy bin 0, angle bin 0
    bins[4] = 1; bins[5] = 2; // P2 in energy bin 1, angle bin 2
    bins[6] = 1; bins[7] = 2; // P3 in energy bin 1, angle bin 2
    pumi_tally->UpdateFilterBins(bins);

    pumi_tally->CopyInitialPositionToBuffer(
        kInitPos.data(), static_cast<Omega_h::LO>(kInitPos.size()));

    // MoveToNextLocation consumes (zeros) flying/weights host arrays,
    // so fresh vectors are needed for each call.
    std::vector<int8_t> flying1(kNumPtcls, 1);
    std::vector<double> weights1(kNumPtcls, 1.0);
    pumi_tally->MoveToNextLocation(
        kInitPos.data(), kDest1.data(), flying1.data(), weights1.data(),
        static_cast<int64_t>(kDest1.size()));

    // Sequential: dest1 → dest2
    std::vector<int8_t> flying2(kNumPtcls, 1);
    std::vector<double> weights2(kNumPtcls, 1.0);
    pumi_tally->MoveToNextLocation(
        kDest1.data(), kDest2.data(), flying2.data(), weights2.data(),
        static_cast<int64_t>(kDest2.size()));

    require_valid_elements(*pumi_tally, "Step 5");
    require_positions_match(pumi_tally->pumipic_ptcls.get(), kDest2, "Step 5");

    const auto &h = pumi_tally->p_pumi_particle_at_elem_boundary_handler;
    auto tallies_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                            h->element_tallies);
    double total_flux = 0.0;
    for (size_t i = 0; i < tallies_host.size(); ++i)
      total_flux += tallies_host(i);
    printf("[INFO] Step 5 multi-dim flux (6 elems × 6 bins = %d entries): "
           "%.6f (expected ~1.3856)\n",
           static_cast<int>(tallies_host.size()), total_flux);
    REQUIRE_THAT(total_flux, Catch::Matchers::WithinAbs(expected_total_flux, 1e-6));

    // Check if the tallies are in correct bins
    // Total in energy bin 0, angle bin 0 and energy bin 1, angle bin 2
    // should each be half of the total flux and they should be equal.
    // tallies_host is DynRankView<double, HostSpace> with rank 3 (nelem, 2, 3)
    const auto nelem = pumi_tally->full_mesh.nelems();
    {
      printf("Multi-dimensional tally (energy x angle):\n");
      printf("\t\t\t  Angle 0  Angle 1  Angle 2  \t\t  Angle 0  Angle 1  Angle 2  \n");
      for (size_t i = 0; i < nelem; ++i) {
        printf("Element %zu:\t", i);
        for (size_t j = 0; j < 2; ++j) {
          printf("Energy %zu: ", j);
          for (size_t k = 0; k < 3; ++k) {
            printf("%f ", tallies_host(i, j, k));
          }
          printf("\t");
        }
        printf("\n");
      }
    }
    printf("\n");
    printf("total flux: %f (expected ~1.3856)\n", total_flux);

    // Check flux in bins:
    // 1. (all, 0, 0) = 1 x 4 x sqrt(3*0.1*0.1) ≈ 0.6928
    double total_all_0_0 = 0.0;
    for (size_t e = 0; e < nelem; ++e) {
      total_all_0_0 += tallies_host(e, 0, 0);
    }
    REQUIRE_THAT(total_all_0_0, Catch::Matchers::WithinAbs(expected_flux_all_0_0, 1e-6));
    // 2. (all, 1, 2) = 1 x 4 x sqrt(3*0.1*0.1) ≈ 0.6928
    double total_all_1_2 = 0.0;
    for (size_t e = 0; e < nelem; ++e) {
      total_all_1_2 += tallies_host(e, 1, 2);
    }
    REQUIRE_THAT(total_all_1_2, Catch::Matchers::WithinAbs(expected_flux_all_1_2, 1e-6));

    printf("[PASS] Step 5: Two sequential moves complete, multi-dim tally "
           "accumulated\n");
  }

  // ================================================================
  // Step 6 — Three sequential moves
  //
  // Cumulative displacement: (+0.30,+0.30,+0.30)
  // Total flux = 3 x 4 x sqrt(3*0.1*0.1) ≈ 2.0785
  
  // ================================================================
  printf("\n===== Step 6: Three sequential moves =====\n");
  {
    const double expected_total_flux = 3 * 4 * std::sqrt(3*0.1*0.1);

    
    auto pumi_tally = std::make_unique<pumitally::PumiTallyImpl>(
        kMeshFilename, kNumPtcls, argc, argv);

    pumi_tally->AddElementTally({1});
    std::vector<unsigned int> bins(kNumPtcls, 0);
    pumi_tally->UpdateFilterBins(bins);

    pumi_tally->CopyInitialPositionToBuffer(
        kInitPos.data(), static_cast<Omega_h::LO>(kInitPos.size()));

    // Fresh flying/weights for each call (consumed by MoveToNextLocation)
    std::vector<int8_t> fly1(kNumPtcls, 1);
    std::vector<double> wgt1(kNumPtcls, 1.0);
    pumi_tally->MoveToNextLocation(
        kInitPos.data(), kDest1.data(), fly1.data(), wgt1.data(),
        static_cast<int64_t>(kDest1.size()));

    std::vector<int8_t> fly2(kNumPtcls, 1);
    std::vector<double> wgt2(kNumPtcls, 1.0);
    pumi_tally->MoveToNextLocation(
        kDest1.data(), kDest2.data(), fly2.data(), wgt2.data(),
        static_cast<int64_t>(kDest2.size()));

    std::vector<int8_t> fly3(kNumPtcls, 1);
    std::vector<double> wgt3(kNumPtcls, 1.0);
    pumi_tally->MoveToNextLocation(
        kDest2.data(), kDest3.data(), fly3.data(), wgt3.data(),
        static_cast<int64_t>(kDest3.size()));

    require_valid_elements(*pumi_tally, "Step 6");
    require_positions_match(pumi_tally->pumipic_ptcls.get(), kDest3, "Step 6");

    const auto &h = pumi_tally->p_pumi_particle_at_elem_boundary_handler;
    auto tallies_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                            h->element_tallies);
    double total_flux = 0.0;
    for (size_t i = 0; i < tallies_host.size(); ++i)
      total_flux += tallies_host(i);
    printf("[INFO] Step 6 total flux: %.6f (expected ~2.0785)\n", total_flux);
    REQUIRE_THAT(total_flux, Catch::Matchers::WithinAbs(expected_total_flux, 1e-6));

    printf("[PASS] Step 6: Three sequential moves complete\n");
  }

  // ================================================================
  // Step 7 — Reflective boundary condition with particle reflection
  //
  // Reflects the remaining displacement (dest - inter_point), matching
  // OpenMC's behavior where the particle's direction of travel is
  // reflected and the remaining path length is conserved.
  //
  //   P0: (0.90,0.90,0.90) → (1.10,1.10,1.10)  flying=1 (all inside)
  //   P1: (1.70,0.90,0.90) → (2.10,0.90,0.90)  flying=1 (x>2, reflects)
  //   P2: (0.90,1.70,0.90) → (0.90,1.70,0.90)  flying=0 (stays)
  //   P3: (0.90,0.90,1.70) → (0.90,1.30,2.10)  flying=1 (z>2, reflects)
  //
  // Hand-calc tally (incident = dest - inter_point):
  //   P0:  |(1.10,1.10,1.10)-(0.90,0.90,0.90)| = √(3·0.04) ≈ 0.346410
  //   P1:  to boundary (2.00,0.90,0.90) → 0.30
  //        incident=(0.10,0,0), n=(-1,0,0) → reflected=(-0.10,0,0)
  //        new dest=(1.90,0.90,0.90), reflected track=0.10  total 0.40
  //   P2:  flying=0 → 0.0
  //   P3:  dir=(0,0.40,0.40), |dir|=0.4√2, hits z=2 at t=0.75
  //        intersection (0.90,1.20,2.00), track to boundary = 0.3√2
  //        incident=(0,0.10,0.10), n=(0,0,-1) → reflected=(0,0.10,-0.10)
  //        new dest=(0.90,1.30,1.90), reflected track = 0.1√2
  //        P3 total = 0.4√2 ≈ 0.565685
  //   Total: 0.2·√3 + 0.4 + 0.4·√2 ≈ 1.312096
  //
  // Expected positions after reflection:
  //   P0: (1.10, 1.10, 1.10)   P1: (1.90, 0.90, 0.90)
  //   P2: (0.90, 1.70, 0.90)   P3: (0.90, 1.30, 1.90)
  // ================================================================
  printf("\n===== Step 7: Reflective boundary condition =====\n");
  {
    const double expected_total_flux =
        0.2 * std::sqrt(3.0) + 0.4 + 0.4 * std::sqrt(2.0);

    auto pumi_tally = std::make_unique<pumitally::PumiTallyImpl>(
        kMeshFilename, kNumPtcls, argc, argv);

    pumi_tally->AddElementTally({1});
    std::vector<unsigned int> bins(kNumPtcls, 0);
    pumi_tally->UpdateFilterBins(bins);

    // Enable reflective boundary condition before the move
    pumi_tally->SetReflectiveBoundaryCondition();

    const auto &h = pumi_tally->p_pumi_particle_at_elem_boundary_handler;
    REQUIRE(h->boundary_condition ==
            pumitally::ParticleAtElemBoundary::BoundaryCondition::REFLECTIVE);

    // Start from kDest3 positions (after three previous moves)
    pumi_tally->CopyInitialPositionToBuffer(
        kDest3.data(), static_cast<Omega_h::LO>(kDest3.size()));

    // Move 4 destinations: two particles cross domain boundaries
    // clang-format off
    std::vector<double> kDest4 = {
        1.10, 1.10, 1.10,  // P0: inside domain, flying=1
        2.10, 0.90, 0.90,  // P1: x>2, flying=1 (reflects at x=2)
        0.90, 1.70, 0.90,  // P2: stays, flying=0
        0.90, 1.30, 2.10   // P3: z>2, flying=1 (reflects at z=2)
    };
    // clang-format on
    std::vector<int8_t> flying4 = {1, 1, 0, 1};
    std::vector<double> weights4(kNumPtcls, 1.0);
    pumi_tally->MoveToNextLocation(
        kDest3.data(), kDest4.data(), flying4.data(), weights4.data(),
        static_cast<int64_t>(kDest4.size()));

    require_valid_elements(*pumi_tally, "Step 7");

    // Expected positions after reflection (OpenMC-style: remaining displacement):
    //   P1: intersection (2.00,0.90,0.90), incident=dest-inter=(0.10,0,0)
    //       n=(-1,0,0) → reflected=(-0.10,0,0)
    //       new dest = (2.00,0.90,0.90)+(-0.10,0,0) = (1.90,0.90,0.90)
    //   P3: intersection (0.90,1.20,2.00), incident=dest-inter=(0,0.10,0.10)
    //       n=(0,0,-1) → reflected=(0,0.10,-0.10)
    //       new dest = (0.90,1.20,2.00)+(0,0.10,-0.10) = (0.90,1.30,1.90)
    // clang-format off
    std::vector<double> kExpectedPos = {
        1.10, 1.10, 1.10,  // P0
        1.90, 0.90, 0.90,  // P1 — reflected
        0.90, 1.70, 0.90,  // P2 — stayed
        0.90, 1.30, 1.90   // P3 — reflected
    };
    // clang-format on
    require_positions_match(pumi_tally->pumipic_ptcls.get(), kExpectedPos,
                            "Step 7");

    // Verify total flux matches hand calculation
    auto tallies_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                            h->element_tallies);
    double total_flux = 0.0;
    for (size_t i = 0; i < tallies_host.size(); ++i)
      total_flux += tallies_host(i);
    printf("[INFO] Step 7 total flux: %.6f (expected ~%.6f)\n",
           total_flux, expected_total_flux);
    REQUIRE_THAT(total_flux,
                 Catch::Matchers::WithinAbs(expected_total_flux, 1e-6));

    printf("[PASS] Step 7: Reflective BC — particles reflected, tally "
           "correct\n");
  }

  // ================================================================
  // Step 8 — Full workflow: construct → tally → move → write Tallies
  //
  // Exercises the complete lifecycle with different start positions,
  // element + node tallies, multiple moves, and Tallies output.
  // ================================================================
  printf("\n===== Step 8: Full workflow — tally, move, write Tallies =====\n");
  {
    auto pumi_tally = std::make_unique<pumitally::PumiTallyImpl>(
        kMeshFilename, kNumPtcls, argc, argv);

    pumi_tally->AddElementTally({2, 3}); // 2 energy × 3 angle
    pumi_tally->AddNodeTally({2});       // 2 energy bins at nodes

    std::vector<unsigned int> filter_bins(kNumPtcls * 2, 0);
    pumi_tally->UpdateFilterBins(filter_bins);

    // clang-format off
    std::vector<double> start = {
        0.50, 0.50, 0.50,   1.00, 0.50, 0.50,
        0.50, 1.00, 0.50,   0.50, 0.50, 1.00
    };
    std::vector<double> step1 = {
        0.70, 0.70, 0.50,   1.20, 0.70, 0.50,
        0.70, 1.20, 0.50,   0.70, 0.70, 1.00
    };
    std::vector<double> step2 = {
        0.90, 0.90, 0.70,   1.40, 0.90, 0.70,
        0.90, 1.40, 0.70,   0.90, 0.90, 1.20
    };
    // clang-format on

    pumi_tally->CopyInitialPositionToBuffer(
        start.data(), static_cast<Omega_h::LO>(start.size()));

    std::vector<int8_t> fly1(kNumPtcls, 1);
    std::vector<double> wgt1(kNumPtcls, 1.0);
    pumi_tally->MoveToNextLocation(
        start.data(), step1.data(), fly1.data(), wgt1.data(),
        static_cast<int64_t>(step1.size()));

    std::vector<int8_t> fly2(kNumPtcls, 1);
    std::vector<double> wgt2(kNumPtcls, 1.0);
    pumi_tally->MoveToNextLocation(
        step1.data(), step2.data(), fly2.data(), wgt2.data(),
        static_cast<int64_t>(step2.size()));

    require_valid_elements(*pumi_tally, "Step 8");

    // Write tally results to VTK
    pumi_tally->WriteTallyResults();

    printf("[PASS] Step 8: Full workflow complete, tallies written to VTK\n");
  }

  printf("\n===== ALL STEPS PASSED =====\n");
}
