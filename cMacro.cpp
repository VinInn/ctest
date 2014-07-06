/*
#define PYTHON_WRAPPER(_class,_name) \
  namespace { typedef cond::PayLoadInspector< _class > PythonWrapper;} \
BOOST_PYTHON_MODULE(plugin ## _name ## PyInterface) { \
  class_<PythonWrapper>("Object",init<>()) \
    .def(init<cond::IOVElement>()) \
    .def("print",&PythonWrapper::print) \
    .def("summary",&PythonWrapper::summary); \
} \
namespace { const char * pluginName_="plugin"  #_name "PyInterface"; }\
PYTHON_ID(PythonWrapper::Class, pluginName_)
*/

#define PYTHON_WRAPPER(_class,_name) \
namespace { const char * pluginName_="plugin"  #_name "PyInterface"; }\

PYTHON_WRAPPER(cond::Pedestals,Pedestals);


#include <iostream>
int main() {

  std::cout << pluginName_ << std::endl;
  return 0;
}
