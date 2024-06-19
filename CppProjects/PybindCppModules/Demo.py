
from Pyds  import CppPlugins


modules =CppPlugins.initPyBindLibs()
print( modules['myPybindModule'].licRenderingPybindCPP)
