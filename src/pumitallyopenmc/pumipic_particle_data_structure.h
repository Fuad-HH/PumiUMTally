//
// Created by Fuad Hasan on 12/3/24.
//

#ifndef PUMITALLYOPENMC_PUMIPIC_PARTICLE_DATA_STRUCTURE_H
#define PUMITALLYOPENMC_PUMIPIC_PARTICLE_DATA_STRUCTURE_H


#include <cstdint>
#include <memory>

namespace pumiinopenmc {
    // ------------------------------------------------------------------------------------------------//
    // Struct for PumiTallyImpl
    struct PumiParticleAtElemBoundary;
    struct PumiTallyImpl;


    // PumiTally class
    // This is a pimpl class that contains the implementation of the PumiTallyImpl Struct (Future class)
    class PumiTally {
    public:
        PumiTally(std::string& mesh_filename, int64_t num_particles, int& argc, char**& argv);

        // Public Functions

        // Initialize the particle locations by searching the initial positions of the particles
        // in the mesh for the first step
        void initialize_particle_location(double* init_particle_positions, int64_t size);

        // Get the new destination of the particles and move the particles to the new destination
        void move_to_next_location(double* particle_destinations, int8_t* flying, int64_t size);

        // Normalize the tally and write the tally array to vtk file
        // Future: send the tally to openmc
        void write_pumi_tally_mesh();

        // Destructor
        ~PumiTally();

    private:
        std::unique_ptr<PumiTallyImpl> pimpl;
    };
}


#endif //PUMITALLYOPENMC_PUMIPIC_PARTICLE_DATA_STRUCTURE_H
