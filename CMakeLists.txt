cmake_minimum_required(VERSION 3.20)
project(pumitallyopenmc)
include(FetchContent)
include(GNUInstallDirs)

########################## OPTIONS ###############################
option(BUILD_SHARED_LIBS                    "Build shared libraries"        ON)
option(PUMITALLYOPENMC_ENABLE_TESTS         "Enable tests"                  ON)
option(PUMI_USE_KOKKOS_CUDA                 "Use CUDA"                      OFF)
option(PUMI_MEASURE_TIME                    "Measure time"                  OFF)
###################################################################

############################ CLANG ################################
#set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-extra-arg=-std=c++17")
###################################################################


################################## PUMIPIC ########################
find_package(pumipic REQUIRED)
if(pumipic_FOUND)
    message(STATUS "Found PUMIPIC: ${pumipic_DIR}")
else()
    message(FATAL_ERROR "PUMIPIC not found. Please set pumipic_ROOT")
endif()
###################################################################

####################### Headers and Sources #######################
set(PUMITALLYOPENMC_HEADERS
        src/pumitallyopenmc/pumipic_particle_data_structure.h
)

set(PUMITALLYOPENMC_SOURCES
        src/pumitallyopenmc/pumipic_particle_data_structure.cpp
)


################################## Testing ########################
if(PUMITALLYOPENMC_ENABLE_TESTS)
    message(STATUS "TESTS ENABLED")
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()
###################################################################

################################ RPATH #############################
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True)
file(RELATIVE_PATH relDir
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
)
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relDir})
set(CMAKE_BUILD_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")



################################## Create Library #################
add_library(pumitallyopenmc)
target_sources(pumitallyopenmc PRIVATE ${PUMITALLYOPENMC_SOURCES})
target_link_libraries(pumitallyopenmc PRIVATE pumipic::pumipic)
if(PUMI_USE_KOKKOS_CUDA)
    target_compile_definitions(pumitallyopenmc PRIVATE -DPUMI_USE_KOKKOS_CUDA)
endif()
set_target_properties(pumitallyopenmc PROPERTIES
    PUBLIC_HEADER "${PUMITALLYOPENMC_HEADERS}")

if(PUMI_MEASURE_TIME)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
    if(CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
        message(WARNING "Time measurement won't work properly in Debug mode")
    endif()

    target_compile_definitions(pumitallyopenmc PRIVATE -DPUMI_MEASURE_TIME)
endif()
###################################################################




################################ Install ##########################

target_include_directories(pumitallyopenmc
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src

    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


  # note that it is not CMAKE_INSTALL_PREFIX we are checking here
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(
        STATUS
        "CMAKE_INSTALL_PREFIX is not set\n"
        "Default value: ${CMAKE_INSTALL_PREFIX}\n"
        "Will set it to ${CMAKE_SOURCE_DIR}/install"
    )
    set(CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/install"
        CACHE PATH "Where the library will be installed to" FORCE
    )
else()
    message(
        STATUS
        "CMAKE_INSTALL_PREFIX was already set\n"
        "Current value: ${CMAKE_INSTALL_PREFIX}"
    )
endif()

install(TARGETS pumitallyopenmc
  EXPORT pumitallyopenmcTargets 
  LIBRARY DESTINATION   ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION   ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION   ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pumitallyopenmc
)

install(EXPORT pumitallyopenmcTargets
  FILE pumitallyopenmcTargets.cmake
  NAMESPACE pumitallyopenmc::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pumitallyopenmc
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pumitallyopenmcConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/pumitallyopenmcConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pumitallyopenmc
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/pumitallyopenmcConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pumitallyopenmc
)

###################################################################

