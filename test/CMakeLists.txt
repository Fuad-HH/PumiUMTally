################################## Catch2 ########################
find_package(Catch2 3 QUIET)
if(Catch2_FOUND)
    message(STATUS "Found Catch2: ${Catch2_DIR}")
else()
    message(STATUS "Catch2 not found. Fetching Catch2...")
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.4.0 # or a later release
    )
    FetchContent_MakeAvailable(Catch2)
endif()
###################################################################

##################### Catch2 Integration #####################
include(CTest)
include(Catch)
enable_testing()
##################### Catch2 Integration #####################

############# Test HostWrite Initialization #############
add_executable(test_hostwrite_initialization test_hostwrite_initialization.cpp)
target_link_libraries(test_hostwrite_initialization PRIVATE pumipic::pumipic Catch2::Catch2WithMain)
catch_discover_tests(test_hostwrite_initialization)

add_executable(test_pumi_tally_impl_methods test_pumi_tally_impl_methods.cpp)
target_include_directories(test_pumi_tally_impl_methods PRIVATE ${PROJECT_SOURCE_DIR}/src/pumitallyopenmc)
target_link_libraries(test_pumi_tally_impl_methods PUBLIC Catch2::Catch2WithMain pumipic::pumipic)
catch_discover_tests(test_pumi_tally_impl_methods)
#################### Test HostWrite ####################