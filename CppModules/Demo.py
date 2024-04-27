
from Pyds  import CppPlugins


modules =CppPlugins.init_PYBindLibs()

result = modules['example'].greet2("ss")
print(result)