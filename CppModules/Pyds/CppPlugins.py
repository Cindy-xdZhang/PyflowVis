import os
import sys
import importlib
import re
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

#! try to make the   build, import of pybind module automatic.
def init_PYBindLibs():
    # Add the pyds folder to the module search path
    pyds_path =os.path.dirname(__file__)
    renamingPydFiles(pyds_path)
    sys.path.append(pyds_path)
    # Create a dictionary to store the imported modules
    modules = {}
    # Iterate over all .pyd files in the pyds folder
    for file in os.listdir(pyds_path):
        if file.endswith('.pyd'):
            # Get the module name (without the extension)
            module_name = os.path.splitext(file)[0]
            
            # Dynamically import the module
            module = importlib.import_module(module_name)
            
            # Store the module in the dictionary with the module name as the key
            modules[module_name] = module
    return modules


if __name__ == '__main__':
    init_PYBindLibs()
 