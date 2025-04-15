//
// Created by Fuad Hasan on 3/22/25.
//
#include "MultiGroupXS.h"
#include <H5Cpp.h>
#include <filesystem>

void read_material(MultiGroupXS &mgxs, const H5::H5File &file, int id);

std::vector<std::string> getH5GroupNames(const H5::H5File &file, const std::string &parentName = "/") {
    std::vector<std::string> groupNames;

    H5::Group group = file.openGroup(parentName);
    hsize_t numGroups = group.getNumObjs();
    for (hsize_t i = 0; i < numGroups; i++) {
        if (group.childObjType(i) == H5O_TYPE_GROUP) {
            std::string groupName = group.getObjnameByIdx(i);
            printf("Group name: %s\n", groupName.c_str());
            groupNames.push_back(groupName);
        }
    }
    return groupNames;
}

double readScalarDataset(const H5::H5File &file, const std::string &datasetName) {
    H5::DataSet dataset = file.openDataSet(datasetName);
    H5::DataType dtype = dataset.getDataType();

    double value;
    dataset.read(&value, dtype);

    return value;
}

int64_t readNumberOfGroups(const H5::H5File &file) {
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

bool readFissionable(const H5::H5File &file, const std::string &group_path) {
    H5::Group group = file.openGroup(group_path);
    H5::Attribute attr = group.openAttribute("fissionable");
    H5::DataType dtype = attr.getDataType();

    int8_t value;
    attr.read(dtype, &value);

    return value == 1;
}

int64_t readOrder(const H5::H5File &file, const std::string &group_path) {
    H5::Group group = file.openGroup(group_path);
    H5::Attribute attr = group.openAttribute("order");
    H5::DataType dtype = attr.getDataType();

    int64_t value;
    attr.read(dtype, &value);

    return value;
}

std::string readStringAttribute(const H5::H5File &file, const std::string &group_path, const std::string &attr_name) {
    H5::Group group = file.openGroup(group_path);
    H5::Attribute attr = group.openAttribute(attr_name);
    H5::DataType dtype = attr.getDataType();

    std::string value(attr.getStorageSize(), '\0');
    attr.read(dtype, &value[0]);

    return value;
}


std::vector<std::string> readTemperatures(const H5::H5File &file, const std::string &materialPath) {
    auto temperatures = getH5GroupNames(file, materialPath);
    // remove kTs from the list of temperatures
    temperatures.erase(std::remove(temperatures.begin(), temperatures.end(), "kTs"), temperatures.end());

    // assert all temperatures are valid (check if they are in the format "numK")
    for (const auto &temp: temperatures) {
        if (temp.find("K") == std::string::npos) {
            throw std::runtime_error("Invalid temperature format: " + temp);
        }
    }
    return temperatures;
}

template<typename T>
Omega_h::Write<T> read_xs_T(std::string matName, std::string temp, std::string xs_name, const H5::H5File &file) {
    std::string path = "/" + matName + "/" + temp + "/" + xs_name;
    H5::DataSet dataset = file.openDataSet(path);
    H5::DataType dtype = dataset.getDataType();
    H5::DataSpace dataspace = dataset.getSpace();

    hsize_t dims[1];
    dataspace.getSimpleExtentDims(dims, nullptr);

    auto host_data = std::vector<T>(dims[0]);
    printf("Dim of %s: %d\n", path.c_str(), dims[0]);
    dataset.read(host_data.data(), dtype);

    auto host_oh_data = Omega_h::HostWrite<T>(host_data.size());
    for (int i = 0; i < host_data.size(); i++) {
        printf("%s: %f\n", path.c_str(), host_data[i]);
        host_oh_data[i] = host_data[i];
    }
    return {host_oh_data};
}


MultiGroupXS read_mgxs(std::string &filename) {
    MultiGroupXS mgxs;
    mgxs.sourceFileName = filename;

    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("File " + filename + " does not exist.");
    }
    H5::H5File file(filename, H5F_ACC_RDONLY);

    mgxs.nEnergyGroups = readNumberOfGroups(file);
    printf("Number of groups: %d\n", mgxs.nEnergyGroups);
    mgxs.energyGroupEdges = readGroupEdges(file);
    printf("Found %d group edges\n", mgxs.energyGroupEdges.size());
    assert(mgxs.nEnergyGroups == mgxs.energyGroupEdges.size() - 1);
    printf("Group edges: ");
    auto groupEdges = Omega_h::HostRead<Omega_h::Real>(mgxs.energyGroupEdges);
    for (int i = 0; i < mgxs.nEnergyGroups + 1; i++) {
        auto edge = groupEdges[i];
        printf("%f ", edge);
    }
    printf("\n");

    printf("Reading materials...\n");
    mgxs.materialNames = getH5GroupNames(file);

    for (int i = 0; i < mgxs.materialNames.size(); i++) {
        read_material(mgxs, file, i);
    }


    file.close();
    return mgxs;
}

void read_material(MultiGroupXS &mgxs, const H5::H5File &file, int id) {
    auto matName = mgxs.materialNames[id];
    auto materialName = matName;

    printf("\n\nOpening material %s\n", matName.c_str());
    std::string path = "/" + matName;
    auto isFissionable = readFissionable(file, path);
    printf("Fissionable: %d\n", isFissionable);

    auto order = readOrder(file, path);
    printf("Order: %d\n", order);

    auto representation = readStringAttribute(file, path, "representation");
    printf("Representation: %s\n", representation.c_str());

    auto scatterFormat = readStringAttribute(file, path, "scatter_format");
    printf("Scatter format: %s\n", scatterFormat.c_str());

    auto scatterShape = readStringAttribute(file, path, "scatter_shape");
    printf("Scatter shape: %s\n", scatterShape.c_str());

    printf("Reading temps...\n");
    auto temperatures = readTemperatures(file, path);
    std::vector<CrossSection_T> cross_sections;
    for (auto &temp: temperatures) {
        printf("Temperature: %s\n", temp.c_str());
        auto absorption = read_xs_T<Omega_h::Real>(matName, temp, "absorption", file);
        auto total = read_xs_T<Omega_h::Real>(matName, temp, "total", file);
        auto g_min = read_xs_T<Omega_h::LO>(matName, temp, "scatter_data/g_min", file);
        auto g_max = read_xs_T<Omega_h::LO>(matName, temp, "scatter_data/g_max", file);
        auto scatter = read_xs_T<Omega_h::Real>(matName, temp, "scatter_data/scatter_matrix", file);

        // remove postfix K from temp and convert to double
        double temp_double = std::stod(temp.substr(0, temp.size() - 1));
        cross_sections.emplace_back(CrossSection_T(temp_double, absorption, total, g_min, g_max, scatter));
    }

    // read kTs
    printf("Reading kTs...\n");
    std::vector<double> kTs;
    for (auto &temp: temperatures) {
        //TODO read kTs
    }

    mgxs.materialXSs.emplace_back(
            MaterialXS(matName, isFissionable, int(order), representation, scatterFormat, scatterShape, temperatures, kTs,
                       cross_sections));
}

