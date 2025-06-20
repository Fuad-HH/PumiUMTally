//
// Created by hasanm4 on 6/20/25.
//

#ifndef PUMITALLYOPENMC_DG2CROSSSECTION_H
#define PUMITALLYOPENMC_DG2CROSSSECTION_H

#include <H5Cpp.h>
#include <Omega_h_array.hpp>

class DG2CrossSection {
public:
  // fills the cross-section arrays from the given file
  explicit DG2CrossSection(std::string filename);

  // Getters go here

  // no need to define readers of the cross-section arrays

private:
  std::string sourceFileName_;
  int nEnergyGroups_;
  int nMaterials_;
  int nTemperatures_;

  Kokkos::View<Omega_h::Real ***> sigma_t_;            // mat, T, g
  Kokkos::View<Omega_h::Real ***> sigma_a_;            // mat, T, g
  Kokkos::View<Omega_h::Real ****> scattering_matrix_; // mat, T, g, g
  Kokkos::View<Omega_h::Real ***> sigma_s_;            // mat, T, g
};

#endif // PUMITALLYOPENMC_DG2CROSSSECTION_H
