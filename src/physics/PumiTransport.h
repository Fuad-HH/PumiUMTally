//
// Created by Fuad Hasan on 4/15/25.
//

#ifndef PUMITALLYOPENMC_PUMITRANSPORT_H
#define PUMITALLYOPENMC_PUMITRANSPORT_H


#include "MultiGroupXS.h"
#include <Omega_h_mesh.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Kokkos_Random.hpp>

#define SEED 123

typedef Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>
        random_pool_t;

class SourceGeometry {
public:
    virtual void sampleUniformly(Kokkos::View<double *> location, random_pool_t rpool) = 0;
};

class Sphere : public SourceGeometry {
public:
    Sphere(double radius, double x, double y, double z) : radius(radius), cx(x), cy(y), cz(z) {}

    double radius;
    double cx;
    double cy;
    double cz;

    void sampleUniformly(Kokkos::View<double *> location, random_pool_t rpool) override;
};

class Box : public SourceGeometry {
public:
    Box(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax) : xMin(xMin), xMax(xMax),
                                                                                        yMin(yMin), yMax(yMax),
                                                                                        zMin(zMin), zMax(zMax) {}

    double xMin;
    double xMax;
    double yMin;
    double yMax;
    double zMin;
    double zMax;

    void sampleUniformly(Kokkos::View<double *> location, random_pool_t rpool) override;
};


struct Source {
    std::unique_ptr<SourceGeometry> geometry_p;
    double energy = 0.0;
};

class PumiTransport {
public:
    PumiTransport(std::string meshFile, std::string xsFile, size_t nParticles, int &argc, char **argv,
                  Source source = {std::make_unique<Box>(0, 1, 0, 1, 0, 1), 0},
                  random_pool_t rpool = random_pool_t(SEED));
    void initializeSource();
    void writePositionsForGNUPlot(std::string gnuplotFileName);

    random_pool_t randomPool;
    Source source;
    std::string meshFileName;
    Kokkos::View<int *[2]> materialTemperature; //TODO: for multi-material simulations, it will needed. Now, it's 0,0
    Kokkos::View<double *> particleEnergy;
    Kokkos::View<double *> particleWeight;
    Kokkos::View<double *> particleOrigins;
    Kokkos::View<double *> particleDestinations;

private:

    size_t nParticles_;
    MultiGroupXS mgxs;
};


#endif //PUMITALLYOPENMC_PUMITRANSPORT_H
