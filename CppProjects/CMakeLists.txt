cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 20)

if(UNIX AND NOT APPLE)
set(VCPKG_ROOT "/home/zhanx0o/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT})
endif()



project(FlowGenerator LANGUAGES CXX)




set(sources 
src/VastistasVelocityGenerator.cpp 
include/VastistasVelocityGenerator.h
include/VectorFieldCompute.h
include/commonUtils.h
src/VectorFieldCompute.cpp
src/transformation.cpp
src/transformation.h
src/flowGenerator.cpp
src/flowGenerator.h
src/commonUtils.cpp
src/main.cpp
src/ParalleLICrendering.cpp
src/ParalleLICrendering.h
)

add_executable(FlowGenerator ${sources})
target_include_directories(FlowGenerator PUBLIC include/)
target_include_directories(FlowGenerator PUBLIC external/)
target_include_directories(FlowGenerator PUBLIC external/cereal/include)
target_include_directories(FlowGenerator PUBLIC external/eigen-3.3.7)
target_include_directories(FlowGenerator PUBLIC external/magic_enum/include)
if(UNIX AND NOT APPLE)
    find_package(TBB REQUIRED)
    message("-- linking to tbb.")
    target_link_libraries(FlowGenerator PRIVATE TBB::tbb)
    add_compile_options(-w)
endif()