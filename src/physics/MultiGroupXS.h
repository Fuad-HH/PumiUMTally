//
// Created by Fuad Hasan on 3/17/25.
//

#ifndef PUMITALLYOPENMC_MULTIGROUPXS_H
#define PUMITALLYOPENMC_MULTIGROUPXS_H

#include<Omega_h_array.hpp>
#include<array>
#include<map>

struct TemperatureXSData;

class Material {
public:

private:
    std::vector<TemperatureXSData> temperatureXSData_;
};

class MultiGroupXS {
public:
    int nGroups() const { return n_groups_; }
    bool validGroup(int group) const { return group >= 0 && group < n_groups_; }

    [[nodiscard]]
    std::array<double,2> getGroupBoundaries(int group) const {
        // check if group is valid
        if (!validGroup(group)) {
            std::string message = "Invalid group number " + std::to_string(group) + ". Must be between 0 and " + std::to_string(n_groups_);
            throw std::runtime_error(message);
        }
        return {groupBoundaries_[group], groupBoundaries_[group]};
    }


private:
    int n_groups_;
    Omega_h::Write<Omega_h::Real> groupBoundaries_;

    std::vector<Material> materials_;
};

#endif //PUMITALLYOPENMC_MULTIGROUPXS_H
