cmake_minimum_required(VERSION 3.4...3.18)
project(exposeCpp2Python LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

set(MODULE_NAME CppLicRenderingModule)

add_subdirectory(pybind11)

set(Libsources 
../include/VectorFieldCompute.h 
../src/VectorFieldCompute.cpp 
)


pybind11_add_module(${MODULE_NAME} example.cpp LicRendering.cpp  ${Libsources})


target_include_directories(${MODULE_NAME}  PRIVATE "../include")
# target_link_libraries(${MODULE_NAME} PRIVATE other_lib)



