import os
import sys
import importlib
import re
import datetime
import platform

def renamingPydFiles(pyds_path):
    for file in os.listdir(pyds_path):
        if file.endswith('.pyd'):
      
            match = re.match(r'(.+?)\.cp\d+-win_amd64\.pyd$', file)
            if match:
       
                prefix = match.group(1)
                new_name = prefix + '.pyd'
                
     
                old_path = os.path.join(pyds_path, file)
                new_path = os.path.join(pyds_path, new_name)
                if os.path.exists(new_path):
                   os.remove(new_path)
                os.rename(old_path, new_path)
                print(f"Renamed '{file}' to '{new_name}'")


def rebuildPyBindLibs(module_root_path,libFile, libRelatedFileList):
    """Check CPP file modification date and rebuild the pybind modules if necessary"""
    lib_modification_date = datetime.datetime.fromtimestamp(os.path.getmtime(libFile))

    for file in libRelatedFileList:
        file_modification_date = datetime.datetime.fromtimestamp(os.path.getmtime(file))
        if file_modification_date > lib_modification_date:
            print(f"Rebuilding pybind modules due to changes in {file}")
            os.chdir(module_root_path)
            # parent_folder =os.getcwd()
            os.system(".\\buildCppModules.bat")
            # lib_modification_date = file_modification_date
            break

def initPyBindLibs(ExcludingList=[]):
    current_platform = platform.system()
    if current_platform == 'Windows':
        platform_subfolder = "win32_64"
    elif current_platform == 'Linux':
        platform_subfolder = "linux"
    else:
        raise Exception(f"Unsupported platform: {current_platform}")

    pyds_path =os.path.dirname(__file__)
    module_root_path=os.path.join(pyds_path,platform_subfolder)
    sys.path.append(module_root_path)
    # Create a dictionary to store the imported modules
    modules = {}
    # Iterate over all .pyd files in the pyds folder
    for file in os.listdir(module_root_path):
        if file.endswith('.pyd') or  file.endswith('.so') :
            # Get the module name (without the extension)
            module_name = os.path.splitext(file)[0]
            if module_name in ExcludingList:
                continue
            # Dynamically import the module
            module = importlib.import_module(module_name)
            print(f"Loaded module '{module_name}' from {module_root_path}, version: {module.__version__}")
            
            # Store the module in the dictionary with the module name as the key
            modules[module_name] = module
    return modules


cppMoudules=initPyBindLibs()

if __name__ == '__main__':
    initPyBindLibs()
 