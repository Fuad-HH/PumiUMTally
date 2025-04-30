//
// Created by hasanm4 on 4/15/25.
//

#include "PumiTransport.h"
#include <fstream>

PumiTransport::PumiTransport(std::string meshFile, std::string xsFile,
                             size_t nParticles, int &argc, char **argv,
                             Source source, random_pool_t rpool)
    : mgxs(xsFile), meshFileName(meshFile), nParticles_(nParticles),
      randomPool(rpool), source(std::move(source)) {

  matTempEg =
      Kokkos::View<int *[3]>("materialTemperatureEnergygroup", nParticles_);
  particleEnergy = Kokkos::View<double *>("particleEnergy", nParticles_);
  particleEnergyGroup = Kokkos::View<int *>("particleEnergyGroup", nParticles_);
  particleWeight = Kokkos::View<double *>("particleWeight", nParticles_);
  particlePosition =
      Kokkos::View<double *>("particlePosition", nParticles_ * 3);
  particleDirection =
      Kokkos::View<double *>("particleDirections", nParticles_ * 3);
}

void Sphere::sampleUniformPositions(Kokkos::View<double *> location,
                                    random_pool_t rpool) {
  auto R = radius;
  double center[3] = {cx, cy, cz};
  Kokkos::parallel_for(
      "sample sphere", location.size() / 3, KOKKOS_LAMBDA(const int i) {
        auto generator = rpool.get_state();
        double x = generator.drand(-1, 1);
        double y = generator.drand(-1, 1);
        double z = generator.drand(-1, 1);
        double norm = Kokkos::sqrt(x * x + y * y + z * z);
        if (norm != 0.0f) {
          x /= norm;
          y /= norm;
          z /= norm;
        }
        double r = generator.drand(0, R);
        r = R * Kokkos::cbrt(r);
        rpool.free_state(generator);

        location(3 * i) = center[0] + r * x;
        location(3 * i + 1) = center[1] + r * y;
        location(3 * i + 2) = center[2] + r * z;
      });
}

void Box::sampleUniformPositions(Kokkos::View<double *> location,
                                 random_pool_t rpool) {
  double min[3] = {xMin, yMin, zMin};
  double max[3] = {xMax, yMax, zMax};

  Kokkos::parallel_for(
      "sample box", location.size() / 3, KOKKOS_LAMBDA(const int i) {
        auto generator = rpool.get_state();
        double x = generator.drand(min[0], max[0]);
        double y = generator.drand(min[1], max[1]);
        double z = generator.drand(min[2], max[2]);
        rpool.free_state(generator);

        location(3 * i) = x;
        location(3 * i + 1) = y;
        location(3 * i + 2) = z;
      });
}

void PumiTransport::initializeSource() {
  source.geometry_p->sampleUniformPositions(particlePosition, randomPool);
  auto E = source.energy;
  auto particleWeight_l = particleWeight;
  auto particleEnergy_l = particleEnergy;
  auto nparticles = nParticles_;
  Kokkos::parallel_for(
      "init E and W", nparticles, KOKKOS_LAMBDA(const int i) {
        particleWeight_l(i) = 1.0;
        particleEnergy_l(i) = E;
      });
  mgxs.findEnergyGroupIndex(particleEnergy_l, particleEnergyGroup);

  auto particle_direction_l = particleDirection;
  sampleInitialUniformDirection(particle_direction_l, randomPool);
}

void PumiTransport::writePositionsForGNUPlot(std::string gnuplotFileName) {

  std::ofstream gnuplotFile(gnuplotFileName);
  if (!gnuplotFile.is_open()) {
    std::cerr << "Error opening file: " << gnuplotFileName << std::endl;
    return;
  }

  auto particle_origins_host = Kokkos::create_mirror_view(particlePosition);
  Kokkos::deep_copy(particle_origins_host, particlePosition);

  for (size_t i = 0; i < nParticles_; ++i) {
    gnuplotFile << particle_origins_host(3 * i) << " "
                << particle_origins_host(3 * i + 1) << " "
                << particle_origins_host(3 * i + 2) << "\n";
  }

  gnuplotFile.close();
}

KOKKOS_FUNCTION
void sampleUnformDirection(random_pool_t rpool, double direction[]) {
  // fill in the direction vector
  auto generator = rpool.get_state();
  double x = generator.drand(-1, 1);
  double y = generator.drand(-1, 1);
  double z = generator.drand(-1, 1);
  rpool.free_state(generator);

  const double norm = Kokkos::sqrt(x * x + y * y + z * z);
  if (norm != 0.0f) {
    x /= norm;
    y /= norm;
    z /= norm;
  }
  direction[0] = x;
  direction[1] = y;
  direction[2] = z;
}

void sampleInitialUniformDirection(Kokkos::View<double *> direction,
                                   random_pool_t rpool) {
  Kokkos::parallel_for(
      "sample uniform direction", direction.size() / 3,
      KOKKOS_LAMBDA(const int i) {
        double dir[3];
        sampleUnformDirection(rpool, dir);
        direction(3 * i) = dir[0];
        direction(3 * i + 1) = dir[1];
        direction(3 * i + 2) = dir[2];
      });
}

void PumiTransport::nextCollision() {
  auto weights_l = particleWeight;
  auto energies_l = particleEnergy;
  auto positions_l = particlePosition;
  auto directions_l = particleDirection;
  auto material_l = matTempEg;

  auto sigma_t_l = mgxs.getSigmaT();
  auto sigma_s_l = mgxs.getSigmaS();
  auto sigma_a_l = mgxs.getSigmaA();
  auto scatter_l = mgxs.getScatteringMatrix();

  // update weights
  Kokkos::parallel_for(
      "update weights", nParticles_, KOKKOS_LAMBDA(const int i) {
        auto matid = material_l(i, 0);
        auto temp = material_l(i, 1);
        weights_l(i) *= .1;
      });

  // scatter
  // update energy
  // update position
}
