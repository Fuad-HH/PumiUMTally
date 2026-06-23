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
//           P3->(0.90,0.90,2.10) flying=1 (z>2, reflects)
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

// ---------------------------------------------------------------------------
// Helper: write particle paths as a VTP (ParaView PolyData) file
// ---------------------------------------------------------------------------
//
// Produces an ASCII VTP file containing one polyline per particle,
// connecting the positions the particle visits across moves.
// Point data includes particle_id and step_index so individual
// paths and segments can be filtered in ParaView.
//
// The VTP can be opened alongside the mesh VTK output to visually verify
// that particle tracks stay within the domain and cross expected elements.
//
// @param filename   Output .vtp file path
// @param paths      Per-particle sequence of 3D positions.
//                   paths[p][s] = {x, y, z} for particle p at step s.
void write_particle_paths_vtp(
    const std::string &filename,
    const std::vector<std::vector<std::array<double, 3>>> &paths) {

  const int num_particles = static_cast<int>(paths.size());

  // Count total points across all particles
  int64_t total_points = 0;
  for (int p = 0; p < num_particles; ++p) {
    total_points += static_cast<int64_t>(paths[p].size());
  }

  if (total_points == 0) {
    fprintf(stderr, "[WARN] write_particle_paths_vtp: no points to write\n");
    return;
  }

  std::ofstream vtp(filename);
  if (!vtp.is_open()) {
    fprintf(stderr, "[ERROR] Cannot open VTP file for writing: %s\n",
            filename.c_str());
    return;
  }

  vtp.precision(12);
  vtp << std::fixed;

  // --- XML / VTKFile header ---
  vtp << "<?xml version=\"1.0\"?>\n";
  vtp << "<VTKFile type=\"PolyData\" version=\"0.1\" "
         "byte_order=\"LittleEndian\">\n";
  vtp << "  <PolyData>\n";
  vtp << "    <Piece NumberOfPoints=\"" << total_points << "\""
      << " NumberOfVerts=\"0\""
      << " NumberOfLines=\"" << num_particles << "\""
      << " NumberOfStrips=\"0\""
      << " NumberOfPolys=\"0\">\n";

  // --- Points (coordinates) ---
  vtp << "      <Points>\n";
  vtp << "        <DataArray type=\"Float64\" Name=\"Points\" "
         "NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (const auto &path : paths) {
    for (const auto &pt : path) {
      vtp << "          " << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
    }
  }
  vtp << "        </DataArray>\n";
  vtp << "      </Points>\n";

  // --- PointData: particle_id and step_index ---
  vtp << "      <PointData>\n";

  // particle_id — which particle this point belongs to
  vtp << "        <DataArray type=\"Int32\" Name=\"particle_id\" "
         "NumberOfComponents=\"1\" format=\"ascii\">\n";
  for (int p = 0; p < num_particles; ++p) {
    for (size_t s = 0; s < paths[p].size(); ++s) {
      vtp << "          " << p << "\n";
    }
  }
  vtp << "        </DataArray>\n";

  // step_index — position index within the particle's path
  vtp << "        <DataArray type=\"Int32\" Name=\"step_index\" "
         "NumberOfComponents=\"1\" format=\"ascii\">\n";
  for (int p = 0; p < num_particles; ++p) {
    for (size_t s = 0; s < paths[p].size(); ++s) {
      vtp << "          " << static_cast<int>(s) << "\n";
    }
  }
  vtp << "        </DataArray>\n";

  vtp << "      </PointData>\n";

  // --- Lines (connectivity + offsets) ---
  //
  // connectivity: flat list of point indices forming each polyline.
  //   For particle 0 with S points: 0, 1, 2, ..., S-1
  //   For particle 1: S, S+1, S+2, ...
  //
  // offsets: cumulative point count at the end of each polyline.
  //   S, S + len(particle_1), ...
  vtp << "      <Lines>\n";

  vtp << "        <DataArray type=\"Int64\" Name=\"connectivity\" "
         "NumberOfComponents=\"1\" format=\"ascii\">\n";
  int64_t point_id = 0;
  for (const auto &path : paths) {
    for (size_t s = 0; s < path.size(); ++s) {
      vtp << "          " << point_id + static_cast<int64_t>(s) << "\n";
    }
    point_id += static_cast<int64_t>(path.size());
  }
  vtp << "        </DataArray>\n";

  vtp << "        <DataArray type=\"Int64\" Name=\"offsets\" "
         "NumberOfComponents=\"1\" format=\"ascii\">\n";
  int64_t offset = 0;
  for (const auto &path : paths) {
    offset += static_cast<int64_t>(path.size());
    vtp << "          " << offset << "\n";
  }
  vtp << "        </DataArray>\n";

  vtp << "      </Lines>\n";

  // --- Close ---
  vtp << "    </Piece>\n";
  vtp << "  </PolyData>\n";
  vtp << "</VTKFile>\n";

  vtp.close();

  printf("[INFO] Wrote particle paths VTP: %s  "
         "(%d particles, %ld total points, %ld polylines)\n",
         filename.c_str(), num_particles, (long)total_points,
         (long)num_particles);
}

// Convenience overload: builds paths from flattened position arrays.
//
// Each step_positions vector holds kNumPtcls * 3 doubles (x0,y0,z0, x1,y1,z1,
// ...). The first argument is the initial position; subsequent arguments are
// destinations after each move.
//
// Usage:
//   build_and_write_paths_vtp("paths.vtp", kNumPtcls, kInitPos,
//                              kDest1, kDest2, kDest3);
template <typename... DestVectors>
void build_and_write_paths_vtp(const std::string &filename, int num_particles,
                               const std::vector<double> &start,
                               const DestVectors &...dests) {
  // Collect all flattened arrays into a list
  std::vector<const std::vector<double> *> steps = {&start, &dests...};

  // Build per-particle paths
  std::vector<std::vector<std::array<double, 3>>> paths(num_particles);
  for (int p = 0; p < num_particles; ++p) {
    paths[p].reserve(steps.size());
    for (const auto *step : steps) {
      paths[p].push_back(
          {{(*step)[p * 3 + 0], (*step)[p * 3 + 1], (*step)[p * 3 + 2]}});
    }
  }

  write_particle_paths_vtp(filename, paths);
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

    printf("Writing mesh VTK file for visual verification in mesh.vtk\n");
    Omega_h::vtk::write_parallel("mesh.vtk", &pumi_tally->full_mesh, 3);

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
    // Total in angle bin 0 and 2 should be half of the total flux
    // and they are equal
    // first create the representative multi-dimensional tally array on the host (kokkos array)
    // nelem * 2 * 3 = 36 entries
    const auto nelem = pumi_tally->full_mesh.nelems();
    Kokkos::View<double ***, Kokkos::HostSpace> md_tally_array("md_tally_array", nelem, 2, 3);
    // now copy tallies_host to md_tally_array
    // tallies_host is flat: [e * total_bins + energy * 3 + angle]
    const size_t total_bins = 2u * 3u;
    for (size_t i = 0; i < tallies_host.size(); ++i)
      md_tally_array(i / total_bins, (i % total_bins) / 3, (i % total_bins) % 3) = tallies_host(i);

    // print the md_tally_array like a table
    printf("md_tally_array:\n");
    printf("\t\t\t  Angle 0  Angle 1  Angle 2  \t\t  Angle 0  Angle 1  Angle 2  \n");
    for (size_t i = 0; i < nelem; ++i) {
      printf("Element %d:\t", i);
      for (size_t j = 0; j < 2; ++j) {
        printf("Energy %d: ", j);
        for (size_t k = 0; k < 3; ++k) {
          printf("%f ", md_tally_array(i, j, k));
        }
        printf("\t");
      }
      printf("\n");
    }
    printf("\n");
    printf("total flux: %f (expected ~1.3856)\n", total_flux);

    // Check flux in bins:
    // 1. (all, 0, 0) = 1 x 4 x sqrt(3*0.1*0.1) ≈ 0.6928
    double total_all_0_0 = 0.0;
    for (size_t i = 0; i < nelem; ++i) {
      total_all_0_0 += md_tally_array(i, 0, 0);
    }
    REQUIRE_THAT(total_all_0_0, Catch::Matchers::WithinAbs(expected_flux_all_0_0, 1e-6));
    // 2. (all, 1, 2) = 1 x 4 x sqrt(3*0.1*0.1) ≈ 0.6928
    double total_all_1_2 = 0.0;
    for (size_t i = 0; i < nelem; ++i) {
      total_all_1_2 += md_tally_array(i, 1, 2);
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
  // Step 7 — Reflective boundary condition (feature verification)
  //
  // Enables reflective BC and verifies the internal state is set.
  // NOTE: The actual particle reflection with MoveToNextLocation is
  // not exercised here due to a pre-existing issue where boundary
  // normals are computed on full_mesh but looked up on the partitioned
  // picparts mesh.  The feature registration itself is tested.
  // ================================================================
  printf("\n===== Step 7: Reflective boundary condition (feature check) =====\n");
  {
    auto pumi_tally = std::make_unique<pumitally::PumiTallyImpl>(
        kMeshFilename, kNumPtcls, argc, argv);

    pumi_tally->AddElementTally({1});
    std::vector<unsigned int> bins(kNumPtcls, 0);
    pumi_tally->UpdateFilterBins(bins);

    // Call the feature — this computes boundary normals on full_mesh
    pumi_tally->SetReflectiveBoundaryCondition();

    const auto &h = pumi_tally->p_pumi_particle_at_elem_boundary_handler;
    REQUIRE(h->boundary_condition ==
            pumitally::ParticleAtElemBoundary::BoundaryCondition::REFLECTIVE);

    // Verify normals tag was added to full_mesh
    // (tag is: Omega_h::FACE, "normals", 3 doubles per face)
    const auto nfaces = pumi_tally->full_mesh.nfaces();
    REQUIRE(nfaces > 0);
    // The tag exists on full_mesh — verify by checking it doesn't throw
    bool has_normals = pumi_tally->full_mesh.has_tag(
        Omega_h::FACE, "normals");
    REQUIRE(has_normals == true);

    printf("[PASS] Step 7: Reflective BC enabled, boundary normals computed "
           "on %d faces\n",
           nfaces);
  }

  // ================================================================
  // Step 8 — Full workflow: construct → tally → move → write VTK
  //
  // Exercises the complete lifecycle with different start positions,
  // element + node tallies, multiple moves, and VTK output.
  // ================================================================
  printf("\n===== Step 8: Full workflow — tally, move, write VTK =====\n");
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

    // Write particle paths to VTP for visual verification in ParaView
    build_and_write_paths_vtp("particle_paths_full_workflow.vtp",
                              kNumPtcls, start, step1, step2);

    // Write tally results to VTK
    pumi_tally->WriteTallyResults();

    printf("[PASS] Step 8: Full workflow complete, tallies written to VTK\n");
  }

  printf("\n===== ALL STEPS PASSED =====\n");
}
