//
// Created by Fuad Hasan on 3/17/25.
//

#include "MultiGroupXS.h"
#include <string>
#include <H5Cpp.h>


struct TemperatureData {
    double temperature;
    std::string tempStr;

    Omega_h::Write<Omega_h::Real> sigma_t;
    Omega_h::Write<Omega_h::Real> sigma_a;
    Omega_h::Write<Omega_h::Real> sigma_f;

    // scattering matrix - csr
};



