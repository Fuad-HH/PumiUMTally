//
// Created by Fuad Hasan on 3/17/25.
//

#ifndef PUMITALLYOPENMC_MULTIGROUPXS_H
#define PUMITALLYOPENMC_MULTIGROUPXS_H

#include<Omega_h_array.hpp>
#include<map>
#include <Omega_h_vector.hpp>

class CrossSection_T{
public:
    CrossSection_T(Omega_h::Real temp, Omega_h::Write<Omega_h::Real> abs,
                   Omega_h::Write<Omega_h::Real> total, Omega_h::Write<Omega_h::LO> g_max,
                   Omega_h::Write<Omega_h::LO> g_min, Omega_h::Write<Omega_h::Real> scatter_matrix)
        : temperature(temp), absorption(abs), total(total), g_max(g_max), g_min(g_min),
          scatter_matrix(scatter_matrix) {}



    Omega_h::Real temperature;
    Omega_h::Write<Omega_h::Real> absorption;
    Omega_h::Write<Omega_h::Real> total;

    Omega_h::Write<Omega_h::LO> g_max;
    Omega_h::Write<Omega_h::LO> g_min;
    Omega_h::Write<Omega_h::Real> scatter_matrix;
};

class MaterialXS{
public:
    MaterialXS(std::string name, bool fissionable, int order, std::string representation,
               std::string scatterFormat, std::string scatterShape, std::vector<std::string> temperatures,
               std::vector<double> KTs, std::vector<CrossSection_T> crossSections)
        : materialName(name), isFissionable(fissionable), order(order),
          representation(representation), scatterFormat(scatterFormat),
          scatterShape(scatterShape), temperatures(temperatures), kTs(KTs),
            crossSections(crossSections) {}


    std::string materialName;
    bool isFissionable;
    int order;
    std::string representation;
    std::string scatterFormat;
    std::string scatterShape;

    std::vector<std::string> temperatures;
    std::vector<double> kTs;
    std::vector<CrossSection_T> crossSections;
};


class MultiGroupXS {
public:
    std::string sourceFileName;
    int nEnergyGroups;


    Omega_h::Write<Omega_h::Real> energyGroupEdges;
    std::vector<std::string> materialNames;
    std::vector<MaterialXS> materialXSs;
};


MultiGroupXS read_mgxs(std::string& filename);

#endif //PUMITALLYOPENMC_MULTIGROUPXS_H
