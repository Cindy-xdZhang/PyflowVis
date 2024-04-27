#include <pybind11/pybind11.h>
#include <string>

std::string greet2(const std::string &name)
{
    return "Hello22332, " + name + "!";
}

namespace py = pybind11;
PYBIND11_MODULE(example, m)
{
    m.doc() = "pybind11 example module";
    m.def("greet2", &greet2, "A function that greets a person");
}