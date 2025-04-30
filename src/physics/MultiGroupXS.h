//
// Created by Fuad Hasan on 3/17/25.
//

#ifndef PUMITALLYOPENMC_MULTIGROUPXS_H
#define PUMITALLYOPENMC_MULTIGROUPXS_H

#include <H5Cpp.h>
#include <Omega_h_array.hpp>
#include <Omega_h_vector.hpp>

class MultiGroupXS {
public:
  explicit MultiGroupXS(std::string filename);

  void print();
  std::string getSourceFileName() { return sourceFileName_; }
  int getNumEnergyGroups() const { return nEnergyGroups_; }
  int getNumMaterials() const { return nMaterials_; }
  int getNumTemps() const { return nTemps_; }
  int getOrder() const { return order_; }

  [[nodiscard]] std::vector<std::string> getMaterialNames() {
    return materialNames_;
  }
  [[nodiscard]] std::vector<std::string> getTemperatureNames() {
    return temperatureNames_;
  }
  [[nodiscard]] Omega_h::Read<Omega_h::Real> getEnergyGroupEdges() {
    return energyGroupEdges_;
  }

  [[nodiscard]] Kokkos::View<Omega_h::Real ***> getSigmaT() { return sigma_t_; }
  [[nodiscard]] Kokkos::View<Omega_h::Real ***> getSigmaA() { return sigma_a_; }
  [[nodiscard]] Kokkos::View<Omega_h::Real ***> getSigmaS() { return sigma_s_; }
  [[nodiscard]] Kokkos::View<Omega_h::Real ****> getScatteringMatrix() {
    return scattering_matrix_;
  }

  void rowSumScatteringMatrix();
  void normalizeScatteringMatrix();

private:
  std::string sourceFileName_;
  int nEnergyGroups_;
  int nMaterials_;
  int nTemps_;
  int order_;

  std::vector<std::string> materialNames_;
  std::vector<std::string> temperatureNames_;

  Omega_h::Write<Omega_h::Real> energyGroupEdges_;
  // Omega_h::Write<Omega_h::Real> temperatures_;

  Kokkos::View<Omega_h::Real **> kTs_;                 // mat, T
  Kokkos::View<Omega_h::Real ***> sigma_t_;            // mat, T, g
  Kokkos::View<Omega_h::Real ***> sigma_a_;            // mat, T, g
  Kokkos::View<Omega_h::Real ****> scattering_matrix_; // mat, T, g, g
  Kokkos::View<Omega_h::Real ***> sigma_s_;            // mat, T, g

  void fillXS(const H5::H5File &file);
};

#endif // PUMITALLYOPENMC_MULTIGROUPXS_H
