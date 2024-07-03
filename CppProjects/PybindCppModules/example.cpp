#include <iostream>
#include <pybind11/pybind11.h>
#include <string>

std::string greet2(const std::string& name)
{
    return "Date 2024 05 07 Hello22332xxxxxxxxxx, " + name + "!";
}

class Animal {
public:
    virtual ~Animal() { }
    virtual std::string go(int n_times) = 0;
};

class Dog : public Animal {
public:
    std::string go(int n_times) override
    {
        std::string result;
        for (int i = 0; i < n_times; ++i)
            result += "woof xxxxxx2022! ";
        std::cout << "c+= cout: " << result << std::endl;
        return result;
    }
};
std::string call_go(Animal* animal)
{
    return animal->go(3);
}

namespace py = pybind11;
PYBIND11_MODULE(example, m)
{
    // module metadata
    m.doc() = "pybind11  module expose c++ lic rendering to python.";
    m.attr("__version__") = "0.0.2-stable_noise_lic";

    // module export symbols
    m.def("greet2", &greet2, "A function that greets a person");

    py::class_<Animal>(m, "Animal")
        .def("go", &Animal::go);

    py::class_<Dog, Animal>(m, "Dog")
        .def(py::init<>());

    m.def("call_go", &call_go);
}