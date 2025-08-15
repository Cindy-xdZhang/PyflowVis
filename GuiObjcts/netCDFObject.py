from GuiObjcts.VertexArrayObject import *
from .VisualizationEngine import getEngine
from FLowUtils.netCDFLoader import *
def path2name(path):
    name = path.split('/')[-1].split('\\')[-1]  # Handle both forward and back slashes
    if '.' in name:  # Remove extension if present
        name = name.rsplit('.', 1)[0]
    return name

class NetCDFLoaderOBJ(Object):
    def __init__(self):
        super().__init__("NetCDF")
        self.create_variable_gui("time_step_begin",-1,False, {'widget': 'input'})
        self.create_variable_gui("time_step_end",-1,False)
        # Update the path variable to use a file dialog widget
        self.create_variable_gui("path", "", False, {'widget': 'file_dialog'})
        self.addAction("load cdf file", lambda x:self.loadCDF()) 
        self.create_variable_gui("save_path", "", False, {'widget': 'file_dialog'})
        self.addAction("save cdf file", lambda x:self.saveCDF()) 

    def loadCDF(self):
        resolved_path=self.getValue("path")
        time_step_begin=self.getValue("time_step_begin")
        time_step_end=self.getValue("time_step_end")
        vectorfield=NetCDFLoader.load_vector_field2d(resolved_path,time_step_begin,time_step_end)
        scene=self.getParentScene()
        # Extract name from path - get last folder/file name
        name=path2name(resolved_path)
        scene.getObject("ActiveField").insertField(name,vectorfield)


    def saveCDF(self):
        resolved_path=self.getValue("save_path")
        time_step_begin=self.getValue("time_step_begin")
        time_step_end=self.getValue("time_step_end")
        vector_field:UnsteadyVectorField2D = self.getParentScene().getObject("ActiveField").getActiveField()  
        NetCDFLoader.save_vector_field2d(resolved_path,vector_field,timestep_begin=time_step_begin,timestep_end=time_step_end)
        
            
    