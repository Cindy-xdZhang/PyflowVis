import os
import sys
import importlib
import re
import datetime


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
    # Add the pyds folder to the module search path

    pyds_path =os.path.dirname(__file__)
    module_root_path = os.path.dirname(pyds_path)
    # libBinary=os.path.join(pyds_path, "example.pyd")
    # libFiles=[ "example.cpp"]
    # libFiles=[  os.path.join(module_root_path, file) for file in libFiles]
    # rebuildPyBindLibs(module_root_path,libBinary,libFiles)
    renamingPydFiles(pyds_path)
    sys.path.append(pyds_path)
    # Create a dictionary to store the imported modules
    modules = {}
    print()
    # Iterate over all .pyd files in the pyds folder
    for file in os.listdir(pyds_path):
        if file.endswith('.pyd'):
            # Get the module name (without the extension)
            module_name = os.path.splitext(file)[0]
            if module_name in ExcludingList:
                continue
            # Dynamically import the module
            module = importlib.import_module(module_name)
            print(f"Loaded module '{module_name}' --version {module.__version__}")
            
            # Store the module in the dictionary with the module name as the key
            modules[module_name] = module
    return modules


if __name__ == '__main__':
    initPyBindLibs()
 