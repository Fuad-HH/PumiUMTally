//
// Created by hasanm4 on 4/15/25.
//

#include "PumiTransport.h"
#include <fstream>

PumiTransport::PumiTransport(std::string meshFile, std::string xsFile, size_t nParticles, int &argc, char **argv,
                             Source source,
                             random_pool_t rpool)
        : mgxs(xsFile), meshFileName(meshFile), nParticles_(nParticles), randomPool(rpool), source(std::move(source)){

    materialTemperature = Kokkos::View<int *[2]> ("materialAndTemperature", nParticles_);
    particleEnergy = Kokkos::View<double *>("particleEnergy", nParticles_);
    particleWeight = Kokkos::View<double *>("particleWeight", nParticles_);
    particleOrigins = Kokkos::View<double *>("particleOrigins", nParticles_ * 3);
    particleDestinations = Kokkos::View<double *>("particleDirections", nParticles_ * 3);
}

void Sphere::sampleUniformly(Kokkos::View<double *> location, random_pool_t rpool) {
    auto R = radius;
    double center[3] = {cx, cy, cz};
    Kokkos::parallel_for("sample sphere", location.size()/3, KOKKOS_LAMBDA(const int i) {
        auto generator = rpool.get_state();
        double x = generator.drand(-1,1);
        double y = generator.drand(-1,1);
        double z = generator.drand(-1,1);
        double norm = Kokkos::sqrt(x*x + y*y + z*z);
        if (norm != 0.0f) {
            x /= norm;
            y /= norm;
            z /= norm;
        }
        double r = generator.drand(0, R);
        r = R*Kokkos::cbrt(r);
        rpool.free_state(generator);

        location(3*i) = center[0] + r*x;
        location(3*i+1) = center[1] + r*y;
        location(3*i+2) = center[2] + r*z;
    });

}

void Box::sampleUniformly(Kokkos::View<double *> location, random_pool_t rpool) {
    double min[3] = {xMin, yMin, zMin};
    double max[3] = {xMax, yMax, zMax};

    Kokkos::parallel_for("sample box", location.size()/3, KOKKOS_LAMBDA(const int i) {
        auto generator = rpool.get_state();
        double x = generator.drand(min[0], max[0]);
        double y = generator.drand(min[1], max[1]);
        double z = generator.drand(min[2], max[2]);
        rpool.free_state(generator);

        location(3*i) = x;
        location(3*i+1) = y;
        location(3*i+2) = z;
    });
}

void PumiTransport::initializeSource() {
    source.geometry_p->sampleUniformly(particleOrigins, randomPool);
    auto E = source.energy;
    auto particleWeight_l = particleWeight;
    auto particleEnergy_l = particleEnergy;
    auto nparticles = nParticles_;
    Kokkos::parallel_for("init E and W", nparticles, KOKKOS_LAMBDA(const int i) {
       particleWeight_l(i) = 1.0;
       particleEnergy_l(i) = E;
    });

}

void PumiTransport::writePositionsForGNUPlot(std::string gnuplotFileName) {

    std::ofstream gnuplotFile(gnuplotFileName);
    if (!gnuplotFile.is_open()) {
        std::cerr << "Error opening file: " << gnuplotFileName << std::endl;
        return;
    }

    auto particle_origins_host = Kokkos::create_mirror_view(particleOrigins);
    Kokkos::deep_copy(particle_origins_host, particleOrigins);

    for (size_t i = 0; i < nParticles_; ++i) {
        gnuplotFile << particle_origins_host(3 * i) << " "
                    << particle_origins_host(3 * i + 1) << " "
                    << particle_origins_host(3 * i + 2) << "\n";
    }

    gnuplotFile.close();
}
