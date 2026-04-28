cmake_minimum_required(VERSION 3.20)
project(pumitally)
include(FetchContent)
include(GNUInstallDirs)

########################## OPTIONS ###############################
option(BUILD_SHARED_LIBS                    "Build shared libraries"        ON)
option(PUMITALLY_ENABLE_TESTS               "Enable tests"                  OFF)
option(PUMITALLY_USE_KOKKOS_CUDA            "Use CUDA"                      OFF)
option(PUMITALLY_MEASURE_TIME               "Measure time"                  OFF)
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
set(PUMITALLY_HEADERS
        src/pumitally/PumiTally.h
        src/pumitally/PumiTallyImpl.h
)

set(PUMITALLY_SOURCES
        src/pumitally/PumiTally.cpp
        src/pumitally/PumiTallyImpl.cpp
)


################################## Testing ########################
if(PUMITALLY_ENABLE_TESTS)
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
add_library(pumitally)
target_sources(pumitally PRIVATE ${PUMITALLY_SOURCES})
target_link_libraries(pumitally PRIVATE pumipic::pumipic)
if(PUMITALLY_USE_KOKKOS_CUDA)
    target_compile_definitions(pumitally PRIVATE -DPUMITALLY_USE_KOKKOS_CUDA)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_options(pumitally PRIVATE -G)
    endif()
endif()
set_target_properties(pumitally PROPERTIES
    PUBLIC_HEADER "${PUMITALLY_HEADERS}")

if(PUMITALLY_MEASURE_TIME)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
    if(CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
        message(WARNING "Time measurement won't work properly in Debug mode")
    endif()

    target_compile_definitions(pumitally PRIVATE -DPUMITALLY_MEASURE_TIME)
endif()
###################################################################




################################ Install ##########################

target_include_directories(pumitally
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

install(TARGETS pumitally
  EXPORT pumitallyTargets
  LIBRARY DESTINATION   ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION   ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION   ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pumitally
)

install(EXPORT pumitallyTargets
  FILE pumitallyTargets.cmake
  NAMESPACE pumitally::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pumitally
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pumitallyConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/pumitallyConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pumitally
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/pumitallyConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pumitally
)

###################################################################

