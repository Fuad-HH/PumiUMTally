//
// Created by hasanm4 on 4/15/25.
//

#ifndef PUMITALLYOPENMC_PUMITRANSPORT_H
#define PUMITALLYOPENMC_PUMITRANSPORT_H

#include "MultiGroupXS.h"
#include <Omega_h_mesh.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>

class PumiTransport {
public:
    PumiTransport(std::string meshFile, std::string xsFile, size_t nParticles, int& argc, char** argv);


private:
    MultiGroupXS mgxs;
    Omega_h::Library lib;
    Omega_h::Mesh mesh;
};


#endif //PUMITALLYOPENMC_PUMITRANSPORT_H
