cmake_minimum_required(VERSION 3.4...3.18)
if(UNIX AND NOT APPLE)
set(VCPKG_ROOT "/home/zhanx0o/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT})
endif()


project(exposeCpp2Python LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

set(MODULE_NAME CppLicRenderingModule)

add_subdirectory(pybind11)

set(Libsources 
../include/VectorFieldCompute.h 
../src/VectorFieldCompute.cpp 
../src/commonUtils.cpp 
)


pybind11_add_module(${MODULE_NAME} example.cpp LicRendering.cpp  ${Libsources})


target_include_directories(${MODULE_NAME}  PRIVATE "../include")
target_include_directories(${MODULE_NAME}  PRIVATE "../external/eigen-3.3.7")
target_include_directories(${MODULE_NAME}  PRIVATE "../external/")

if(UNIX AND NOT APPLE)
    find_package(TBB REQUIRED)
    message("-- linking to tbb.")
    target_link_libraries(${MODULE_NAME} PRIVATE TBB::tbb)
    add_compile_options(-w)
endif()





