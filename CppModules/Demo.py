
from Pyds  import CppPlugins


modules =CppPlugins.initPyBindLibs()

result = modules['example'].greet2("ss")
print(result)

dog=modules['example'].Dog()
res=modules['example'].call_go(dog)
print(res)
