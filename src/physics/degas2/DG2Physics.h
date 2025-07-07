//
// Created by Fuad Hasan on 6/20/25.
//

#ifndef PUMITALLYOPENMC_DG2PHYSICS_H
#define PUMITALLYOPENMC_DG2PHYSICS_H

#define SEED 12345

#include "DG2CrossSection.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

typedef Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>
    random_pool_t;

struct ParticleInfo {
  double position[3];  // Position in space (x, y, z)
  double direction[3]; // Direction vector (unit vector)
  double weight;
  double energy_group; // Energy group *index*
  int particle_index;
};

struct FieldInfo {
  double electron_temperature;
  double ion_temperature;
  double electron_density;
  double ion_density;
  double bulk_flow_velocity[3];
};

class DG2Physics {
public:
  DG2Physics(const std::string &cross_section_file, const int num_particles,
             const int seed = SEED)
      : random_pool(seed), cross_section(cross_section_file) {
    // Initialize particle energy
    particle_energy =
        Kokkos::View<double *>("particle_energy", num_particles);
	//cross_sections = Kokkos::View<double *[2]>("cross_sections", num_particles);
  }

  DG2Physics(const DG2CrossSection cross_section, const int num_particles,
             const int seed = SEED)

      : random_pool(seed), cross_section(cross_section) {
    // Initialize particle energy
    particle_energy =
        Kokkos::View<double *>("particle_energy", num_particles);
	//cross_sections = Kokkos::View<double *[2]>("cross_sections", num_particles);
  }

  KOKKOS_FUNCTION
  double ionization_cross_section(double energy, double e_temperature) const{
	double mp {938.27e6/(3e10*3e10)}; //eV/c^2 = eV*s^2/cm^2
    double particle_velocity_squared {2*energy/mp}; //cm^2/s^2

	//Compute Ionization Cross Section
    double coef2 [9];

    coef2[0] = -3.271396786375e1; coef2[1] = 1.353655609057e1; coef2[2] = -5.739328757388;
    coef2[3] = 1.563154982022; coef2[4] = -2.877056004391e-1; coef2[5] = 3.482559773737e-2;
    coef2[6] = -2.631976175590e-3; coef2[7] = 1.119543953861e-4; coef2[8] = -2.039149852002e-6;

    double lnrate_ion {0}; //cm^2
    for (int i=0; i<9; i++) {
        lnrate_ion += coef2[i]*Kokkos::pow(Kokkos::log(e_temperature),i);
    }
	double sigma_ion = Kokkos::exp(lnrate_ion)/Kokkos::sqrt(particle_velocity_squared);
    return energy;
  }

  KOKKOS_FUNCTION
  double charge_exchange_cross_section(double energy, double ion_temperature) const{
	double mp {938.27e6/(3e10*3e10)}; //eV/c^2 = eV*s^2/cm^2
    double particle_velocity_squared {2*energy/mp}; //cm^2/s^2

    //Compute Charge Exchange Cross Section
    double coef [9][9]; //E index, T index

    coef[0][0] = -1.829079581680e1; coef[0][1] = 2.169137615703e-1; coef[0][2] = 4.307131243894e-2;
    coef[0][3] = -5.754895093075e-4; coef[0][4] = -1.552077120204e-3; coef[0][5] = -1.876800283030e-4;
    coef[0][6] = 1.125490270962e-4; coef[0][7] = -1.238982763007e-5; coef[0][8] = 4.163596197181e-07;

    coef[1][0] = 1.640252721210e-1; coef[1][1] = -1.106722014459e-1; coef[1][2] = 8.948693624917e-3;
    coef[1][3] = 6.062141761233e-3; coef[1][4] = -1.210431587568e-3; coef[1][5] = -4.052878751584e-5;
    coef[1][6] = 2.875900435895e-5; coef[1][7] = -2.616998139678e-6; coef[1][8] = 7.558092849125e-8;

    coef[2][0] = 3.364564509137e-2; coef[2][1] = -1.382158680424e-3; coef[2][2] = -1.209480567154e-02;
    coef[2][3] = 1.075907881928e-3; coef[2][4] = 8.297212635856e-4; coef[2][5] = -1.907025662962e-04;
    coef[2][6] = 1.338839628570e-5; coef[2][7] = -1.171762874107e-7; coef[2][8] = -1.328404104165e-8;

    coef[3][0] = 9.530225559189e-3; coef[3][1] = 7.348786286628e-3; coef[3][2] = -3.675019470470e-4;
    coef[3][3] = -8.119301728339e-4; coef[3][4] = 1.361661816974e-4; coef[3][5] = 1.141663041636e-5;
    coef[3][6] = -4.340802793033e-6; coef[3][7] = 3.517971869029e-7; coef[3][8] = -9.170850253981e-09;

    coef[4][0] = -8.519413589968e-4; coef[4][1] = -6.343059502294e-4; coef[4][2] = 1.039643390686e-3;
    coef[4][3] = 8.911036876068e-6; coef[4][4] = -1.008928628425e-4; coef[4][5] = 1.775681984457e-05;
    coef[4][6] = -7.003521917385e-7; coef[4][7] = -4.928692832866e-8; coef[4][8] = 3.208853883734e-9;

    coef[5][0] = -1.247583860943e-3; coef[5][1] = -1.919569450380e-4; coef[5][2] = -1.553840717902e-4;
    coef[5][3] = 3.175388949811e-5; coef[5][4] = 1.080693990468e-5; coef[5][5] = -3.149286923815e-6;
    coef[5][6] = 2.318308730487e-7; coef[5][7] = 1.756388998863e-10; coef[5][8] = -3.952740758950e-10;

    coef[6][0] = 3.014307545716e-4; coef[6][1] = 4.075019351738e-5; coef[6][2] = 2.670827249272e-6;
    coef[6][3] = -4.515123641755e-6; coef[6][4] = 5.106059413591e-7; coef[6][5] = 3.105491554749e-8;
    coef[6][6] = -6.030983538280e-9; coef[6][7] = -1.446756795654e-10; coef[6][8] = 2.739558475782e-11;

    coef[7][0] = -2.499323170044e-5; coef[7][1] = -2.850044983009e-6; coef[7][2] = 7.695300597935e-7;
    coef[7][3] = 2.187439283954e-7; coef[7][4] = -1.299275586093e-7; coef[7][5] = 2.274394089017e-8;
    coef[7][6] = -1.755944926274e-9; coef[7][7] = 7.143183138281e-11; coef[7][8] = -1.693040208927e-12;

    coef[8][0] = 6.932627237765e-7; coef[8][1] = 6.966822400446e-8; coef[8][2] = -3.783302281524e-8;
    coef[8][3] = -2.911233951880e-9; coef[8][4] = 5.117133050290e-9; coef[8][5] = -1.130988250912e-9;
    coef[8][6] = 1.005189187279e-10; coef[8][7] = -3.989884105603e-12; coef[8][8] = 6.388219930167e-14;

    double lnrate_cx {0};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            lnrate_cx += coef[i][j]*Kokkos::pow(Kokkos::log(energy),i)*Kokkos::pow(Kokkos::log(ion_temperature),j);
        }
    }
    double sigma_cx = Kokkos::exp(lnrate_cx)/Kokkos::sqrt(particle_velocity_squared);
	return sigma_cx;
  }


  KOKKOS_FUNCTION
  void sample_collision_distance(ParticleInfo &particle_info,
                                 const FieldInfo &field_info) const {
	// example of sampling a random number
    auto rand_gen = random_pool.get_state();
    double x = rand_gen.drand(0., 1.);
    random_pool.free_state(rand_gen);

	double energy = particle_energy(particle_info.particle_index);

	double sigma_ion = ionization_cross_section(energy, field_info.electron_temperature);
	double sigma_cx = charge_exchange_cross_section(energy, field_info.ion_temperature);

	//Generate distance and move particle
    double l =-Kokkos::log(x)/(field_info.electron_density*sigma_ion+field_info.ion_density*sigma_cx); //cm. n in cm^-3
    particle_info.position[0] += l*particle_info.direction[0];
    particle_info.position[1] += l*particle_info.direction[1];
    particle_info.position[2] += l*particle_info.direction[2];
  }

  // collision event
  KOKKOS_FUNCTION
  void collide_particle(ParticleInfo &particle_info,
                        const FieldInfo &field_info) const {

	//Generate Random Numbers
	auto rand_gen = random_pool.get_state();
    double x1 = rand_gen.drand(0., 1.);
	double x2 = rand_gen.drand(0., 1.);
	double y1 = rand_gen.drand(0., 1.);
	double y2 = rand_gen.drand(0., 1.);
	double ww = rand_gen.drand(0., 1.);
    random_pool.free_state(rand_gen);

	double energy = particle_energy(particle_info.particle_index);

	double sigma_ion = ionization_cross_section(energy, field_info.electron_temperature);
	double sigma_cx = charge_exchange_cross_section(energy, field_info.ion_temperature);

	//Compute New Direction and Energy and set particle info
	//Compute 3 Maxwellian (Gaussian) distributed velocities (cm/s)
	 double mp {938.27e6/(3e10*3e10)}; //eV/c^2 = eV*s^2/cm^2

	auto vx = Kokkos::sqrt(field_info.ion_temperature/mp)*Kokkos::sqrt(-2 * Kokkos::log(x1))*Kokkos::cos(2*M_PI*x2);
    auto vy = Kokkos::sqrt(field_info.ion_temperature/mp)*Kokkos::sqrt(-2 * Kokkos::log(x1))*Kokkos::sin(2*M_PI*x2);
	auto vz = Kokkos::sqrt(field_info.ion_temperature/mp)*Kokkos::sqrt(-2 * Kokkos::log(y1))*Kokkos::sin(2*M_PI*y2);

	auto mag_v = Kokkos::sqrt(vx*vx + vy*vy + vz*vz);

	particle_info.direction[0] = vx/mag_v;
	particle_info.direction[1] = vy/mag_v;
	particle_info.direction[2] = vz/mag_v;

	particle_energy(particle_info.particle_index) = 0.5*mp*mag_v*mag_v;
	particle_info.energy_group = particle_energy(particle_info.particle_index); //Temporary for debugging
	//Adjust Weights
	double new_weight = particle_info.weight*(1-
        (field_info.electron_density*sigma_ion)/
		(field_info.ion_density*sigma_cx+field_info.electron_density*sigma_ion));

	//Russian Roulette
	double wc = 0.25;
	double ws = 1.0;

	if (new_weight < wc){
        if (ww<(1-new_weight/ws)) {
            new_weight = 0;
        }
        else {
            new_weight = ws;
        }
    }
	//particle_info.weight = new_weight;
	particle_info.weight = sigma_ion;

}
//To here
  random_pool_t random_pool;
  DG2CrossSection cross_section;

  Kokkos::View<double *> particle_energy;

};
#endif // PUMITALLYOPENMC_DG2PHYSICS_H
