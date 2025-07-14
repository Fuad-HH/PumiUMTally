/*
 * This is created as a proof of concept for the Degas2 Neutral Transport Code
 * Provides parallel search capabilities and tallying to degas2 through this
 * PUMI-Tally interface
 */

#include "DG2Physics.h"
#include <pumipic_particle_data_structure.h>

// ******************************************* Helper Functions
// ******************************************* //
void read_input_parameters(int argc, char *const *argv, std::string &mesh_name,
                           int &num_particles);
void print_initial_info(const std::string &mesh_name,
                        int num_particles) noexcept;

int main(int argc, char *argv[]) {

  std::string mesh_name;
  int num_particles;
  read_input_parameters(argc, argv, mesh_name, num_particles);
  print_initial_info(mesh_name, num_particles);

  // Initialize PUMI-Tally
  auto pumitally =
      pumiinopenmc::PumiTally(mesh_name, num_particles, argc, argv);

  return 0;
}

// ****************************************************************************************************************
// //
// ********************************************* Function Definitions
// ********************************************* //
// ****************************************************************************************************************
// //

void print_initial_info(const std::string &mesh_name,
                        int num_particles) noexcept {
  printf("\n===================> Accelerated Degas2 <======================\n");
  std::string mesh_particle_info =
      "\tMesh: " + mesh_name +
      "\n\tNumber of particles: " + std::to_string(num_particles) + "\n";
  printf("%s", mesh_particle_info.c_str());
  printf("\n===============================================================\n");
}

void read_input_parameters(int argc, char *const *argv, std::string &mesh_name,
                           int &num_particles) {
  if (argc != 3) {
    throw std::runtime_error("Usage: " + std::string(argv[0]) +
                             " <mesh_name> <num_particles>");
  }
  mesh_name = argv[1];
  num_particles = std::stoi(argv[2]);
  if (num_particles <= 0) {
    throw std::runtime_error("Number of particles must be a positive integer.");
  }
}
