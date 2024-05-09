cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 20)
project(FlowGenerator LANGUAGES CXX)




set(sources 
src/VastistasVelocityGenerator.cpp 
src/flowGenerator.cpp
)

add_executable(FlowGenerator ${sources})
target_include_directories(FlowGenerator PUBLIC include/)