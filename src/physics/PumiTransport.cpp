//
// Created by hasanm4 on 4/15/25.
//

#include "PumiTransport.h"

PumiTransport::PumiTransport(std::string meshFile, std::string xsFile, size_t nParticles, int& argc, char** argv){
    lib = Omega_h::Library(&argc, &argv);
    mesh = Omega_h::binary::read(meshFile, &lib);
}

