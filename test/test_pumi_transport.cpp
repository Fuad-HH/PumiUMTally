//
// Created by Fuad Hasan on 4/29/25.
//

#include "PumiTransport.h"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <catch2/catch_test_macros.hpp>


TEST_CASE("Test Transport") {
    Kokkos::initialize();
    {
        int argc = 1;
        char *argv[1];
        auto defaultBoxSourceTransport = PumiTransport("notused", "assets/mgxs.h5", 10000, argc, argv);
        defaultBoxSourceTransport.initializeSource();
        // TODO: remove this line
        defaultBoxSourceTransport.writePositionsForGNUPlot("../build/test_box.dat");

        auto sphereSourceTransport = PumiTransport("notused", "assets/mgxs.h5", 10000, argc, argv, {std::make_unique<Sphere>(3.0, 1,2,3), 0});
        sphereSourceTransport.initializeSource();
        // TODO: remove this line
        sphereSourceTransport.writePositionsForGNUPlot("../build/test_sphere.dat");
    }
    Kokkos::finalize();

}