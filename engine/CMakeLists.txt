cmake_minimum_required(VERSION 3.22)
project(juniauto_engine
    VERSION 0.1.0
    DESCRIPTION "JuniAuto C++ core engine"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Compile flags ----
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -fvisibility=hidden
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG -march=x86-64-v3)
endif()

# ---- Dependencies ----
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(pybind11 2.13 REQUIRED CONFIG)
find_package(CURL REQUIRED)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(Threads REQUIRED)

# ---- Core static library ----
add_library(juniauto_core STATIC
    src/data/pipeline.cpp
    src/data/features.cpp
    src/data/freshness.cpp
    src/bayesian/regression.cpp
    src/bayesian/posterior.cpp
    src/bayesian/shrinkage.cpp
    src/costs/model.cpp
    src/costs/spread.cpp
    src/costs/slippage.cpp
    src/costs/impact.cpp
    src/gateway/decision.cpp
    src/gateway/friction.cpp
    src/gateway/action.cpp
    src/portfolio/sizing.cpp
    src/portfolio/covariance.cpp
    src/portfolio/constraints.cpp
    src/api/alpaca_client.cpp
    src/utils/math.cpp
    src/utils/stats.cpp
    src/utils/time.cpp
)

target_include_directories(juniauto_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(juniauto_core
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        CURL::libcurl
        nlohmann_json::nlohmann_json
        Threads::Threads
)

# ---- Python module (pybind11) ----
pybind11_add_module(quant_engine MODULE
    src/bindings/pybind_module.cpp
)

target_link_libraries(quant_engine PRIVATE juniauto_core)

# ---- Install ----
install(TARGETS quant_engine
    LIBRARY DESTINATION lib/python
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# ---- Tests (optional) ----
option(JUNIAUTO_BUILD_TESTS "Build engine tests" OFF)
if(JUNIAUTO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
