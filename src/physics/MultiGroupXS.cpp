//
// Created by Fuad Hasan on 3/17/25.
//

#include "MultiGroupXS.h"
#include <Omega_h_for.hpp>
#include <filesystem>
#include <string>

int readOrder(const H5::H5File &file, const std::string &group_path);
bool checkOrderZero(const H5::H5File &file,
                    const std::vector<std::string> &materialNames);
bool checkNonFissionable(const H5::H5File &file,
                         const std::vector<std::string> &materialNames);
void readOtherAttributes(H5::H5File &file,
                         const std::vector<std::string> &materialNames);
int readNumberOfGroups(const H5::H5File &file);
Omega_h::Write<Omega_h::Real> readGroupEdges(const H5::H5File &file);
std::vector<std::string> getH5GroupNames(const H5::H5File &file,
                                         const std::string &parentName = "/");
std::vector<std::string> readTemperatures(const H5::H5File &file,
                                          const std::string &materialPath);
std::vector<std::string>
readTemperatureForAllMaterials(const H5::H5File &file,
                               const std::vector<std::string> &materialNames);

MultiGroupXS::MultiGroupXS(std::string filename) : sourceFileName_(filename) {
  if (!std::filesystem::exists(sourceFileName_)) {
    throw std::runtime_error("File " + sourceFileName_ + " does not exist.");
  }
#ifndef NDEBUG
  printf("[INFO] Reading XS from %s...\n", sourceFileName_.c_str());
#endif
  H5::H5File file(sourceFileName_, H5F_ACC_RDONLY);

  materialNames_ = getH5GroupNames(file, "/");
  nMaterials_ = materialNames_.size();
  if (nMaterials_ == 0) {
    throw std::runtime_error("No materials found in the file.");
  }

  temperatureNames_ = readTemperatureForAllMaterials(file, materialNames_);
  nTemps_ = temperatureNames_.size();
  if (nTemps_ == 0) {
    throw std::runtime_error("No temperatures found in the file.");
  }

  nEnergyGroups_ = readNumberOfGroups(file);
  energyGroupEdges_ = readGroupEdges(file);
  assert(nEnergyGroups_ == energyGroupEdges_.size() - 1);

  assert(checkOrderZero(file, materialNames_));
  assert(checkNonFissionable(file, materialNames_));

#ifndef NDEBUG
  readOtherAttributes(file, materialNames_);
#endif

  kTs_ = Kokkos::View<Omega_h::Real **>("kTs", nMaterials_, nTemps_);
  sigma_t_ = Kokkos::View<Omega_h::Real ***>("sigma_t", nMaterials_, nTemps_,
                                             nEnergyGroups_);
  sigma_a_ = Kokkos::View<Omega_h::Real ***>("sigma_a", nMaterials_, nTemps_,
                                             nEnergyGroups_);
  sigma_s_ = Kokkos::View<Omega_h::Real ***>("sigma_s", nMaterials_, nTemps_,
                                             nEnergyGroups_);
  scattering_matrix_ =
      Kokkos::View<Omega_h::Real ****>("scattering matrix", nMaterials_,
                                       nTemps_, nEnergyGroups_, nEnergyGroups_);

  fillXS(file);
  rowSumScatteringMatrix();
  normalizeScatteringMatrix();
  file.close();
}

void MultiGroupXS::fillXS(const H5::H5File &file) {
  auto kTs_h = Kokkos::create_mirror_view(kTs_);
  auto sigma_t_h = Kokkos::create_mirror_view(sigma_t_);
  auto sigma_a_h = Kokkos::create_mirror_view(sigma_a_);
  auto scattering_matrix_h = Kokkos::create_mirror_view(scattering_matrix_);

  for (int matId = 0; matId < materialNames_.size(); ++matId) {
    const auto &materialName = materialNames_[matId];
    auto materialpath = "/" + materialName;
    auto kTsPath = materialpath + "/kTs";

    /*
    H5::DataSet kTsDataset = file.openDataSet(kTsPath);
    H5::DataSpace kTsDataspace = kTsDataset.getSpace();
    H5::DataType kTsType = kTsDataset.getDataType();
    hsize_t kTsDim[1] = {0};
    kTsDataspace.getSimpleExtentDims(kTsDim, nullptr);
    assert(kTsDim[0] == nTemps_);
    auto kTsData = std::vector<Omega_h::Real>(kTsDim[0]);
    kTsDataset.read(kTsData.data(), kTsType);
    for (int tempId = 0; tempId < nTemps_; ++tempId) {
        kTs_h(matId, tempId) = kTsData[tempId];
    }
     */

    for (int tempId = 0; tempId < temperatureNames_.size(); ++tempId) {
      const auto &temp = temperatureNames_[tempId];
      auto temperaturePath = materialpath + "/" + temp;
      const auto absorptionPath = temperaturePath + "/absorption";
      const auto totalPath = temperaturePath + "/total";
      const auto scatteringPath =
          temperaturePath + "/scatter_data/scatter_matrix";

#ifndef NDEBUG
      printf("Temperature path: %s\n", temperaturePath.c_str());
      printf("Absorption path: %s\n", absorptionPath.c_str());
      printf("Total path: %s\n", totalPath.c_str());
      printf("Scattering path: %s\n", scatteringPath.c_str());
#endif

      H5::DataSet absorptionDataset = file.openDataSet(absorptionPath);
      H5::DataSet totalDataset = file.openDataSet(totalPath);
      H5::DataSet scatteringDataset = file.openDataSet(scatteringPath);
      H5::DataSpace absorptionDataspace = absorptionDataset.getSpace();
      H5::DataSpace totalDataspace = totalDataset.getSpace();
      H5::DataSpace scatteringDataspace = scatteringDataset.getSpace();
      H5::DataType absorptionType = absorptionDataset.getDataType();
      H5::DataType totalType = totalDataset.getDataType();
      H5::DataType scatteringType = scatteringDataset.getDataType();

      hsize_t absorptionDim[1] = {0};
      hsize_t totalDim[1] = {0};
      hsize_t scatteringDim[1] = {0};
      absorptionDataspace.getSimpleExtentDims(absorptionDim, nullptr);
      totalDataspace.getSimpleExtentDims(totalDim, nullptr);
      scatteringDataspace.getSimpleExtentDims(scatteringDim, nullptr);
      assert(absorptionDim[0] == nEnergyGroups_);
      assert(totalDim[0] == nEnergyGroups_);
      assert(scatteringDim[0] == nEnergyGroups_ * nEnergyGroups_);

      auto absorptionData = std::vector<Omega_h::Real>(absorptionDim[0]);
      auto totalData = std::vector<Omega_h::Real>(totalDim[0]);
      auto scatteringData = std::vector<Omega_h::Real>(scatteringDim[0]);
      absorptionDataset.read(absorptionData.data(), absorptionType);
      totalDataset.read(totalData.data(), totalType);
      scatteringDataset.read(scatteringData.data(), scatteringType);

      for (int g = 0; g < nEnergyGroups_; ++g) {
        sigma_t_h(matId, tempId, g) = totalData[g];
        sigma_a_h(matId, tempId, g) = absorptionData[g];

        // Symmetric scattering matrix
        for (int g2 = 0; g2 < nEnergyGroups_; ++g2) {
          scattering_matrix_h(matId, tempId, g, g2) =
              scatteringData[g * nEnergyGroups_ + g2];
        }
      }
    }
  }

  // Copy data from host to device
  Kokkos::deep_copy(kTs_, kTs_h);
  Kokkos::deep_copy(sigma_t_, sigma_t_h);
  Kokkos::deep_copy(sigma_a_, sigma_a_h);
  Kokkos::deep_copy(scattering_matrix_, scattering_matrix_h);
}

void MultiGroupXS::rowSumScatteringMatrix() {
  using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  using member_type = team_policy::member_type;

  auto scattering_matrix_l = scattering_matrix_;
  auto sigma_s_l = sigma_s_;

  Kokkos::parallel_for(
      "row sum sigma_s", team_policy(nMaterials_, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type &team_member) {
        int m = team_member.league_rank(); // material index
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member,
                                    scattering_matrix_l.extent(1) *
                                        scattering_matrix_l.extent(2)),
            [=](const int &i) {
              int t = i / scattering_matrix_l.extent(2); // temperature index
              int g = i % scattering_matrix_l.extent(2); // group index
              Omega_h::Real sum = 0.0;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member,
                                            scattering_matrix_l.extent(2)),
                  [=](int j, double &lsum) {
                    lsum += scattering_matrix_l(m, t, g, j);
                  },
                  sum);
              sigma_s_l(m, t, g) = sum;
            });
      });
}

void MultiGroupXS::normalizeScatteringMatrix() {
  using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  using member_type = team_policy::member_type;
  auto scattering_matrix_l = scattering_matrix_;
  auto sigma_s_l = sigma_s_;
  const int nmat = scattering_matrix_l.extent(0);
  const int ntemp = scattering_matrix_l.extent(1);
  const int ngroup = scattering_matrix_l.extent(2);
  team_policy policy(nmat, Kokkos::AUTO);

  Kokkos::parallel_for(
      "normalize scattering matrix", policy,
      KOKKOS_LAMBDA(const member_type &team) {
        const int m = team.league_rank(); // material index
        for (int t = 0; t < ntemp; ++t) { // temperature index
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, ngroup),
              [&](const int g) { // group index
                double decom = sigma_s_l(m, t, g);
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, ngroup),
                                     [&](const int j) { // group index
                                       scattering_matrix_l(m, t, g, j) /= decom;
                                     });

                // cumulative sum
                team.team_barrier();

                double partial_sum = 0.0;
                Kokkos::parallel_scan(
                    Kokkos::ThreadVectorRange(team, ngroup),
                    [&](const int j, double &sum, const bool final) {
                      sum += scattering_matrix_l(m, t, g, j);
                      if (final) {
                        scattering_matrix_l(m, t, g, j) = sum;
                      }
                    },
                    partial_sum);
              });
        }
      });
}

std::vector<std::string> getH5GroupNames(const H5::H5File &file,
                                         const std::string &parentName) {
  std::vector<std::string> groupNames;

  H5::Group group = file.openGroup(parentName);
  hsize_t numGroups = group.getNumObjs();
  for (hsize_t i = 0; i < numGroups; i++) {
    if (group.childObjType(i) == H5O_TYPE_GROUP) {
      std::string groupName = group.getObjnameByIdx(i);
      groupNames.push_back(groupName);
    }
  }
  return groupNames;
}

std::vector<std::string> readTemperatures(const H5::H5File &file,
                                          const std::string &materialPath) {
  auto temperatures = getH5GroupNames(file, materialPath);
  // remove kTs from the list of temperatures
  temperatures.erase(
      std::remove(temperatures.begin(), temperatures.end(), "kTs"),
      temperatures.end());

  // assert all temperatures are valid (check if they are in the format "numK")
  for (const auto &temp : temperatures) {
    if (temp.find("K") == std::string::npos) {
      throw std::runtime_error("Invalid temperature format: " + temp);
    }
  }
  return temperatures;
}

std::vector<std::string>
readTemperatureForAllMaterials(const H5::H5File &file,
                               const std::vector<std::string> &materialNames) {
  std::vector<std::vector<std::string>> temperatures(materialNames.size());
  for (int i = 0; i < materialNames.size(); i++) {
    temperatures[i] = readTemperatures(file, materialNames[i]);
  }

  // make sure all materials have the same temperatures
  for (int i = 1; i < temperatures.size(); i++) {
    if (temperatures[i] != temperatures[0]) {
      std::string errorMsg =
          "Materials have different temperatures: " + temperatures[i][0] +
          " and " + temperatures[0][0];
      throw std::runtime_error(errorMsg);
    }
  }
  return temperatures[0];
}

int readNumberOfGroups(const H5::H5File &file) {
  H5::Attribute attr = file.openAttribute("energy_groups");
  H5::DataType dtype = attr.getDataType();

  int64_t energy_groups;
  attr.read(dtype, &energy_groups);

  return energy_groups;
}

[[nodiscard]]
Omega_h::Write<Omega_h::Real> readGroupEdges(const H5::H5File &file) {
  H5::Attribute attr = file.openAttribute("group structure");
  H5::DataType dtype = attr.getDataType();
  H5::DataSpace dataspace = attr.getSpace();

  hsize_t dims[1];
  dataspace.getSimpleExtentDims(dims, nullptr);

  auto host_data = std::vector<Omega_h::Real>(dims[0]);
  printf("Dim of group structure: %d\n", dims[0]);
  attr.read(dtype, host_data.data());
  auto host_oh_data = Omega_h::HostWrite<Omega_h::Real>(host_data.size());
  for (int i = 0; i < host_data.size(); i++) {
    host_oh_data[i] = host_data[i];
  }
  return {host_oh_data};
}

int readOrder(const H5::H5File &file, const std::string &group_path) {
  H5::Group group = file.openGroup(group_path);
  H5::Attribute attr = group.openAttribute("order");
  H5::DataType dtype = attr.getDataType();

  int value;
  attr.read(dtype, &value);

  return value;
}

bool checkOrderZero(const H5::H5File &file,
                    const std::vector<std::string> &materialNames) {
  for (const auto &materialName : materialNames) {
    int order = readOrder(file, materialName);
    if (order != 0) {
      printf("[ERROR] Order of material %s is %d, expected 0\n",
             materialName.c_str(), order);
      return false;
    }
  }
  return true;
}

bool readFissionable(const H5::H5File &file, const std::string &group_path) {
  H5::Group group = file.openGroup(group_path);
  H5::Attribute attr = group.openAttribute("fissionable");
  H5::DataType dtype = attr.getDataType();

  int8_t value;
  attr.read(dtype, &value);

  return value == 1;
}

bool checkNonFissionable(const H5::H5File &file,
                         const std::vector<std::string> &materialNames) {
  for (const auto &materialName : materialNames) {
    bool fissionable = readFissionable(file, materialName);
    if (fissionable) {
      printf("[ERROR] Material %s is fissionable, expected non-fissionable\n",
             materialName.c_str());
      return false;
    }
  }
  return true;
}

std::string readStringAttribute(const H5::H5File &file,
                                const std::string &group_path,
                                const std::string &attr_name) {
  H5::Group group = file.openGroup(group_path);
  H5::Attribute attr = group.openAttribute(attr_name);
  H5::DataType dtype = attr.getDataType();

  std::string value(attr.getStorageSize(), '\0');
  attr.read(dtype, &value[0]);

  return value;
}

void readOtherAttributes(H5::H5File &file,
                         const std::vector<std::string> &materialNames) {
  for (const auto &materialName : materialNames) {
    auto path = "/" + materialName;
    auto representation = readStringAttribute(file, path, "representation");
    printf("Representation: %s\n", representation.c_str());

    auto scatterFormat = readStringAttribute(file, path, "scatter_format");
    printf("Scatter format: %s\n", scatterFormat.c_str());

    auto scatterShape = readStringAttribute(file, path, "scatter_shape");
    printf("Scatter shape: %s\n", scatterShape.c_str());
  }
}

void MultiGroupXS::print() {
  printf("\n\n\n\n=>=============== MultiGroupXS ===============<=\n");
  printf("Source file name: %s\n", sourceFileName_.c_str());
  printf("Number of energy groups: %d\n", nEnergyGroups_);
  printf("Number of materials: %d\n", nMaterials_);
  printf("Number of temperatures: %d\n", nTemps_);

  printf("\n\n\n");
  printf("Material names:\n");
  for (const auto &materialName : materialNames_) {
    printf("  %s,", materialName.c_str());
  }
  printf("\n");
  printf("Temperature names:\n");
  for (const auto &temperatureName : temperatureNames_) {
    printf("  %s,", temperatureName.c_str());
  }
  printf("\n");
  printf("Energy group edges:\n");
  auto energyGroupEdges_h =
      Omega_h::HostWrite<Omega_h::Real>(energyGroupEdges_);
  for (int i = 0; i < energyGroupEdges_h.size(); ++i) {
    printf("  %5.3f,", energyGroupEdges_h[i]);
  }
  printf("\n\n\n");

  /*
  printf("kTs:\n");
  auto kTs_h = Kokkos::create_mirror_view(kTs_);
  Kokkos::deep_copy(kTs_h, kTs_);
  for (int i = 0; i < nMaterials_; ++i) {
      for (int j = 0; j < nTemps_; ++j) {
          printf("  %5.3f,", kTs_h(i, j));
      }
      printf("\n");
  }
  printf("\n\n\n");
  */
  printf("Sigma_t:\n");
  auto sigma_t_h = Kokkos::create_mirror_view(sigma_t_);
  Kokkos::deep_copy(sigma_t_h, sigma_t_);
  for (int i = 0; i < nMaterials_; ++i) {
    for (int j = 0; j < nTemps_; ++j) {
      for (int g = 0; g < nEnergyGroups_; ++g) {
        printf("  %5.3f,", sigma_t_h(i, j, g));
      }
      printf("\n");
    }
  }
  printf("\n\n\n");
  printf("Sigma_a:\n");
  auto sigma_a_h = Kokkos::create_mirror_view(sigma_a_);
  Kokkos::deep_copy(sigma_a_h, sigma_a_);
  for (int i = 0; i < nMaterials_; ++i) {
    for (int j = 0; j < nTemps_; ++j) {
      for (int g = 0; g < nEnergyGroups_; ++g) {
        printf("  %5.3f,", sigma_a_h(i, j, g));
      }
      printf("\n");
    }
  }
  printf("\n\n\n");

  printf("Sigma_s:\n");
  auto sigma_s_h = Kokkos::create_mirror_view(sigma_s_);
  Kokkos::deep_copy(sigma_s_h, sigma_s_);
  for (int i = 0; i < nMaterials_; ++i) {
    for (int j = 0; j < nTemps_; ++j) {
      for (int g = 0; g < nEnergyGroups_; ++g) {
        printf("  %5.3f,", sigma_s_h(i, j, g));
      }
      printf("\n");
    }
  }
  printf("\n\n\n");

  printf("Scattering Matrix:\n");
  auto scattering_matrix_h = Kokkos::create_mirror_view(scattering_matrix_);
  Kokkos::deep_copy(scattering_matrix_h, scattering_matrix_);
  for (int i = 0; i < nMaterials_; ++i) {
    for (int j = 0; j < nTemps_; ++j) {
      for (int g = 0; g < nEnergyGroups_; ++g) {
        for (int g2 = 0; g2 < nEnergyGroups_; ++g2) {
          printf("  %5.3f,", scattering_matrix_h(i, j, g, g2));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n\n");
  }

  printf("=>=============== MultiGroupXS ===============<=\n\n\n");
}

void MultiGroupXS::findEnergyGroupIndex(Kokkos::View<double *> energy,
                                        Kokkos::View<int *> group) const {
  auto group_edges_l = energyGroupEdges_;

  Omega_h::parallel_for(
      "findEnergyIndex", group.size(), OMEGA_H_LAMBDA(const int i) {
        double energy_value = energy[i];
        int group_index = -1;
        for (int j = 0; j < group_edges_l.size() - 1; ++j) {
          if (energy_value >= group_edges_l[j] &&
              energy_value < group_edges_l[j + 1]) {
            group_index = j;
            break;
          }
        }
        group[i] = group_index;
      });
}
