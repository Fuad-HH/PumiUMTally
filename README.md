[![Test](https://github.com/Fuad-HH/PumiUMTally/actions/workflows/test.yml/badge.svg)](https://github.com/Fuad-HH/PumiUMTally/actions/workflows/test.yml)
[![Static Analysis](https://github.com/Fuad-HH/PumiUMTally/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/Fuad-HH/PumiUMTally/actions/workflows/static-analysis.yml)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
# PUMI-Tally

**GPU-accelerated unstructured mesh tallies for Monte Carlo particle transport**

PUMI-Tally accelerates unstructured mesh tallies in Monte Carlo neutral particle transport simulations by exploiting mesh adjacency information on CPUs and GPUs. Built on top of [PUMIPic](https://github.com/SCOREC/pumi-pic) and [Kokkos](https://kokkos.org/), it provides distributed parallel particle and mesh data structures with [Omega_h](https://github.com/SCOREC/omega_h).

## Installation

### Option 1: Install with Spack (recommended)

Spack automates the build of PUMI-Tally and all its dependencies.

1. **Create and activate a Spack environment**

```bash
spack env create pumi-tally-env
spack env activate pumi-tally-env
```

2. **Update the builtin Spack repo** (requires at least `releases/v2026.02`)

```bash
spack repo update builtin --branch releases/v2026.02
```

3. **Add the PUMI-PIC Spack repository**

```bash
spack repo add https://github.com/SCOREC/pumi-pic-spack.git --name pumi-pic-spack
```

4. **Add packages and install**

For OpenMC with PUMI-Tally support:

```bash
spack add openmc-pumi ^kokkos+openmp+serial
spack concretize --force
spack install
```
You may not need DAGMC but it installs OpenMC with DAGMC support by default.
Change the DAGMC spec off if you don't need it.

> [!Tip]
> For tallying on the GPUs, use `^kokkos+cuda cuda_arch=<arch_code>` instead. Check `spack info pumi-tally` for the supported architectures.

Verify the installation:

```bash
# If OpenMC was installed
openmc --help          # should show the --ohMesh option

# If only PUMI-Tally was installed
spack find pumi-tally
```

### Option 2: Build from source
PUMI-Tally involves many dependencies and is complex to build from source. A complete build instruction for platform-specific
versions is provided in the [PUMIPic Wiki](https://github.com/SCOREC/pumi-pic/wiki). Use this install PUMIPic. Be sure
to install the [make_search_class](https://github.com/Fuad-HH/pumi-pic/tree/make_search_class) branch of PUMIPic.

#### Build PUMI-Tally

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=$DEPS_DIR \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_INSTALL_PREFIX=$DEPS_DIR \
  -DBUILD_SHARED_LIBS=ON
cmake --build build -j$(nproc) --target install
```

> [!TIP]
> Add `-DPUMI_USE_KOKKOS_CUDA=ON` when building with CUDA.
> Add `-DPUMITALLYOPENMC_ENABLE_TESTS=ON` to build the test suite (requires [Catch2 v3.4.0](https://github.com/catchorg/Catch2)).

#### OpenMC with PUMI-Tally (optional)

PUMI-Tally currently works with a [specific fork of OpenMC](https://github.com/Fuad-HH/openmc/tree/decouple_pumi_tally). HDF5 with MPI and high-level API support is required.

```bash
git clone --recurse-submodules --depth 1 --branch decouple_pumi_tally \
  https://github.com/Fuad-HH/openmc.git /tmp/openmc
cmake -S /tmp/openmc -B /tmp/openmc/build \
  -DCMAKE_INSTALL_PREFIX=$DEPS_DIR \
  -DCMAKE_PREFIX_PATH=$DEPS_DIR \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DOPENMC_USE_MPI=ON \
  -DOPENMC_USE_OPENMP=ON \
  -DOPENMC_USE_PUMIPIC=ON
cmake --build /tmp/openmc/build -j$(nproc) --target install
```
This will install OpenMC without DAGMC support. To enable DAGMC, install DAGMC first using the instructions in [DAGMC Website](https://svalinn.github.io/DAGMC/install/openmc.html).

> [!NOTE]
> This fork uses `--recurse-submodules` because OpenMC vendors some dependencies as git submodules.

## Usage with OpenMC

### Preparing the mesh

Create a first-order (linear) tetrahedral volume mesh of the tally region using any meshing tool (Gmsh, Simmetrix, etc.). Requirements:

- The geometry must be **convex** (no concavity).
- The mesh must cover the entire OpenMC geometry with no internal holes. If the OpenMC geometry contains holes, add a bounding box around it.

Convert the mesh to Omega_h format (`.osh`). For example, from Gmsh:

```bash
msh2osh input.msh output.osh
```

Check and adjust the coordinate scale to match the OpenMC model:

```bash
describe output.osh        # prints coordinate min/max
scale output.osh scaled.osh 10   # scale by 10 in all axes if needed
```

### Running

```bash
openmc --ohMesh mesh.osh
```

Results are written to `fluxresult.vtk`.

## How it works

PUMI-Tally decouples tally operations from OpenMC via the [PIMPL idiom](https://www.geeksforgeeks.org/cpp/pimpl-idiom-in-c-with-examples/), so OpenMC does not need to link against PUMIPic, Kokkos, or any GPU compiler. Transport runs on the CPU or GPU based on the physics application that connects to it
(for example, [OpenMC](https://github.com/openmc-dev/openmc) runs on the CPU); tallies run on the CPU or GPU (however `Kokkos` is compiled) through a batched interface of three calls:

1. **`PumiTally` and `CopyInitialPosition`** — Initializes the PUMI-Tally object and localizes particles to their parent mesh elements based on the physics application's source sampling strategy.
2. **`MoveToNextLocation`** — copies particle destinations, weights, and status from OpenMC to the device, then walks each particle through the mesh element-by-element using adjacency information, accumulating track-length tallies per element via atomics — no dynamic allocations or re-localization trees needed.
3. **`WriteTallyResults`** — writes the accumulated tallies to disk (currently VTK only).



![Detailed explanation of library public methods](images/public_methods_explanation.svg)

*High-level API of PUMI-Tally in OpenMC.*

## Citation

If you use PUMI-Tally in your research, please cite:

```bibtex
@article{hasan2025gpu,
  title   = {GPU Acceleration of Monte Carlo Tallies on Unstructured Meshes in OpenMC with PUMI-Tally},
  author  = {Hasan, Fuad and Smith, Cameron W and Shephard, Mark S and Churchill, R Michael
             and Wilkie, George J and Romano, Paul K and Shriwise, Patrick C and Merson, Jacob S},
  journal = {arXiv preprint arXiv:2504.19048},
  year    = {2025}
}
```

## License

See [LICENSE](LICENSE) for details.
