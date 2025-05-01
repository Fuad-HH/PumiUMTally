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
        particleWeight_l(i) = 1000.0;
        particleEnergy_l(i) = E;
      });
  mgxs.findEnergyGroupIndex(particleEnergy_l, particleEnergyGroup);

  auto particleEnergyGroup_l = particleEnergyGroup;
  auto matTempEg_l = matTempEg;
  Kokkos::parallel_for(
      "copy group", nparticles, KOKKOS_LAMBDA(const int i) {
        matTempEg_l(i, 2) = particleEnergyGroup_l(i);
      });

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

KOKKOS_FUNCTION
void rotate_direction(const double u[], double mu, double phi, double new_u[]) {
  double cos = std::cos(phi);
  double sin = std::sin(phi);
  double sqrt_mu = std::sqrt(1. - mu * mu);
  double sqrt_w = std::sqrt(1. - u[2] * u[2]);

  double ux, uy, uz;

  if (sqrt_w > 1.E-10) {
    ux = mu * u[0] + sqrt_mu * (u[0] * u[2] * cos - u[1] * sin) / sqrt_w;
    uy = mu * u[1] + sqrt_mu * (u[1] * u[2] * cos + u[0] * sin) / sqrt_w;
    uz = mu * u[2] - sqrt_mu * sqrt_w * cos;
  } else {
    double sqrt_v = std::sqrt(1. - u[1] * u[1]);

    ux = mu * u[0] + sqrt_mu * (u[0] * u[1] * cos + u[2] * sin) / sqrt_v;
    uy = mu * u[1] - sqrt_mu * sqrt_v * cos;
    uz = mu * u[2] + sqrt_mu * (u[1] * u[2] * cos - u[0] * sin) / sqrt_v;
  }
  new_u[0] = ux;
  new_u[1] = uy;
  new_u[2] = uz;
}

void PumiTransport::nextCollision(random_pool_t rpool) {
  auto weights_l = particleWeight;
  auto energies_l = particleEnergy;
  auto positions_l = particlePosition;
  auto directions_l = particleDirection;
  auto material_l = matTempEg;
  auto mat_temp_eg_l = matTempEg;

  auto sigma_t_l = mgxs.getSigmaT();
  auto sigma_s_l = mgxs.getSigmaS();
  auto sigma_a_l = mgxs.getSigmaA();
  auto scatter_matrix_l = mgxs.getScatteringMatrix();
  auto group_edges = mgxs.getEnergyGroupEdges();

  // update weights
  Kokkos::parallel_for(
      "update weights", nParticles_, KOKKOS_LAMBDA(const int i) {
        auto matid = material_l(i, 0);
        auto temp = material_l(i, 1);
        auto eg = material_l(i, 2);

        auto absorb = sigma_a_l(matid, temp, eg);
        auto scatter = sigma_s_l(matid, temp, eg);
        auto total = sigma_t_l(matid, temp, eg);

        weights_l(i) = weights_l(i) * (1.0 - (absorb / total));

        // russian roulette
        auto generator = rpool.get_state();
        auto rand = generator.drand(0, 1);
        double p_kill = 1.0 - Kokkos::abs(weights_l(i)) / 0.2; // cutoff 0.2
        weights_l(i) = (rand > p_kill) ? 1000.0 : 0.0; // survival weight 1.0

        // scatter
        auto rand2 = generator.drand(0, 1);
        int scattered_group = -1;
        for (int g = 0; g < scatter_matrix_l.extent(3); ++g) {
          // scattered_group = (rand2 < scatter_matrix_l(matid, temp, eg, g)) ?
          // g : scattered_group;
          if (rand2 < scatter_matrix_l(matid, temp, eg, g)) {
            scattered_group = g;
            break; // Exit the loop once the group is found
          }
        }
        if (scattered_group == -1) {
          printf("[ERROR] No scattered group found\n");
        }
        mat_temp_eg_l(i, 2) = scattered_group;

        // update energy: middle of the group
        auto prev_energy = energies_l(i);
        auto new_energy = 0.5 * (group_edges[mat_temp_eg_l(i, 2)] +
                                 group_edges[mat_temp_eg_l(i, 2) + 1]);
        energies_l(i) = new_energy;

        // scatter direction: calculate cosine of the angle based on change of
        // energy
        double A = 5.0; // TODO: this has to come from the material
        double alpha = ((A - 1) / (A + 1));
        alpha *= alpha;
        // double mu = ((1+alpha)*(1-alpha))/(2*(new_energy/prev_energy));
        double mu = generator.drand(-1, 1); // TODO: resolve physics here
        double phi = generator.drand(0, 1.0) * 2 * M_PI;
        double new_direction[3] = {0.0, 0.0, 0.0};
        rotate_direction(&directions_l(3 * i), mu, phi, new_direction);

        directions_l(3 * i) = new_direction[0];
        directions_l(3 * i + 1) = new_direction[1];
        directions_l(3 * i + 2) = new_direction[2];

        // get distance to next collision
        double distance =
            -Kokkos::log(generator.drand(1e-100, 1)) / total * 100;

        // update next position
        positions_l(3 * i) += distance * directions_l(3 * i);
        positions_l(3 * i + 1) += distance * directions_l(3 * i + 1);
        positions_l(3 * i + 2) += distance * directions_l(3 * i + 2);
        // printf("Sample distance: %f, x: %f, y: %f, z: %f\n", distance,
        //        positions_l(3 * i), positions_l(3 * i + 1),
        //        positions_l(3 * i + 2));

        rpool.free_state(generator);
      });
}

bool PumiTransport::areParticlesAlive() {
  // if all weights are 0, then all particles are dead
  auto particle_weights_l = particleWeight;
  double sum = 0.0;

  Kokkos::parallel_reduce(
      "sum weights", nParticles_,
      KOKKOS_LAMBDA(const int i, double &local_sum) {
        local_sum += particle_weights_l(i);
      },
      sum);
  return sum > 1e-10;
}
