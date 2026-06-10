//
// Created by Fuad Hasan on 6/20/25.
//

#ifndef PUMITALLY_DG2CROSSSECTION_H
#define PUMITALLY_DG2CROSSSECTION_H

#include <Omega_h_array.hpp>

class DG2CrossSection {
public:
  // fills the cross-section arrays from the given file
  explicit DG2CrossSection(std::string filename);

  // explicitly given cross-section arrays constructor
  DG2CrossSection(Kokkos::View<Omega_h::Real ***> sigma_t,
                  Kokkos::View<Omega_h::Real ***> sigma_a,
                  Kokkos::View<Omega_h::Real ****> scattering_matrix,
                  Kokkos::View<Omega_h::Real ***> sigma_s)
      : nEnergyGroups(sigma_t.extent(2)), nMaterials(sigma_t.extent(0)),
        nTemperatures(sigma_t.extent(1)), sigma_t(sigma_t), sigma_a(sigma_a),
        scattering_matrix(scattering_matrix), sigma_s(sigma_s) {}

  DG2CrossSection(const DG2CrossSection &other) = default;

  // no need to define readers of the cross-section arrays

  int nEnergyGroups;
  int nMaterials;
  int nTemperatures;

  Kokkos::View<Omega_h::Real ***> sigma_t;            // mat, T, g
  Kokkos::View<Omega_h::Real ***> sigma_a;            // mat, T, g
  Kokkos::View<Omega_h::Real ****> scattering_matrix; // mat, T, g, g
  Kokkos::View<Omega_h::Real ***> sigma_s;            // mat, T, g
};

#endif // PUMITALLY_DG2CROSSSECTION_H
