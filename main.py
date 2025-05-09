from VisualizationEngine import *
from train import *
from FLowUtils.AnalyticalFlowCreator import *
from DeepUtils.utils import EasyConfig
from VertexArrayObject import *
from GuiObjcts.ObjectGUIReflection import ValueGuiCustomization
from shaderManager import *
from FLowUtils.VectorField2d import *
from FLowUtils.netCDFLoader import *
from NLPCommand import *
from  PlanarManifold import *

def screen_to_world(x, y, width, height, modelview, projection, viewport):    
    y = height - y  # OpenGL's y axis  is reversed of pygame's y axis
    z = gl.glReadPixels(x, y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
    return gluUnProject(x, y, z, modelview, projection, viewport)

class GuiTest(Object):
    def __init__(self):
        super().__init__("GuiTest")
        
        self.create_variable_gui("boolean_var", True, False,{'widget': 'checkbox'})
        self.create_variable_gui("boolean_var_default", True, False)
        self.create_variable_gui("checkbox_int",1,False,{'widget': 'checkbox'})
        self.create_variable_gui("input_int",1,False, {'widget': 'input'})
        self.create_variable_gui("default_int",1,False)
        self.create_variable_gui("slider_float",0.5,False, {'widget': 'slider_float', 'min': 0.0, 'max': 1.0})
        self.create_variable_gui("slider_float",0.5,False, {'widget': 'input'})
        self.create_variable_gui("default_float",0.5,False) 
        self.create_variable_gui("color_vec3", (255.0, 0.0, 0.0), False,{'widget': 'color_picker'})
   
        self.create_variable_gui("drag_ivec3", (255, 0, 0), False,{'widget': 'drag'})
        self.create_variable_gui("default_ivec3", (255, 0, 0), False)
        self.create_variable_gui("color_vec4",[1.0,1.0,1.0,1.0],False)
        self.appendGuiCustomization(ValueGuiCustomization("color_vec4","vec4",{'widget': 'color_picker'}) )
        
        self.create_variable("input_vec4", [1, 1, 1, 1])        
        self.create_variable_gui("default_vec4", (255, 0, 0,0))
        
        self.create_variable_gui("input_ivec3", (255, 0, 0), False,{'widget': 'input'})
        self.create_variable_gui("ivecn", (0, 0, 1,1,0,2))
        self.create_variable_gui("vecn", (255, 0, 0,0,0,0))

        self.create_variable_gui("float_array_var_plot", [0.1, 0.2, 0.3, 0.4,0.2], False,{'widget': 'plot_lines'})         
        self.create_variable_gui("string_var", "Hello ImGui", False,{'widget': 'input'})
        self.create_variable_gui("string_var2", "Hello ImGui", False)
        
        self.addAction("reload NoiseImage", lambda object: print("reload image")) 
        
        testDictionary = { "a": 1, "array0": [0.1, 0.2, 0.3, 0.4,0.2], "StepSize2": 3.0,"sonDictionary":{"son_a": 11, "array1": [0.3, 0.2, 0.3],"gradSondict": {"gradSon_b":22 ,"gradVec":[1,2,3]}}}
        self.create_variable("testDictionary",testDictionary,False)
   
   
        
def path2name(path):
    name = path.split('/')[-1].split('\\')[-1]  # Handle both forward and back slashes
    if '.' in name:  # Remove extension if present
        name = name.rsplit('.', 1)[0]
    return name


class Renderable:
    def __init__(self, image_data,x=0.0,y=0.0,z=0.0):
        self.width, self.height = image_data.shape[1]/10, image_data.shape[0]/10
        self.texture_id = create_texture(image_data)
        self.x = x
        self.y = y
        self.z = z
        self.name="renderable"

   
    def check_collision(self, point):
        half_width, half_height = self.width / 2, self.height / 2
        min_x, max_x = self.x - half_width, self.x+ half_width
        min_y, max_y = self.y- half_height, self.y + half_height
        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
            self.selected = True
            logging.debug(f"click select success")
        else:
            self.selected = False

    def draw(self):
    
        glPushMatrix()  # Save the current matrix state
        glTranslatef(self.x, self.y, self.z)  # Move the object to its position

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(-self.width / 2,  -self.height / 2,0)
        glTexCoord2f(1.0, 0.0); glVertex3f(self.width / 2,-self.height / 2,0)
        glTexCoord2f(1.0, 1.0); glVertex3f(self.width / 2, self.height / 2,0)
        glTexCoord2f(0.0, 1.0); glVertex3f(-self.width / 2,  self.height / 2,0)
        glEnd()
        glPopMatrix()  # Restore the matrix state
        
    def eventCallBacks(self,event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:# mid button
            x, y = event.pos
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            world_coordinates = screen_to_world(x, y, viewport[2], viewport[3], modelview, projection, viewport)
            self.check_collision(world_coordinates[:3])
            

class NetCDFLoaderOBJ(Object):
    def __init__(self):
        super().__init__("NetCDF")
        self.create_variable_gui("time_step_begin",-1,False, {'widget': 'input'})
        self.create_variable_gui("time_step_end",-1,False)
        # Update the path variable to use a file dialog widget
        self.create_variable_gui("path", "", False, {'widget': 'file_dialog'})
        self.addAction("load cdf file", lambda x:self.loadCDF()) 
        
    def loadCDF(self):
        path=self.getValue("path")
      
        vectorfield=NetCDFLoader.load_vector_field2d(path)
        scene=self.getParentScene()
        # Extract name from path - get last folder/file name
        name=path2name(path)
        scene.getObject("ActiveField").insertField(name,vectorfield)
        
            
    
        
        

                    





#! todo: lic
#! todo: pathline
#! todo: reference frame transforamtion of lic and pathlines


def main():
    config = EasyConfig()
    config.load("config/renderingConfig.yaml", recursive=False)
    engine=VisualizationEngine(config=config['rendering'])
    size=config['rendering']["window_size"]

    
    # Assuming you have your LIC texture data ready
    lic_texture_data = np.random.rand(100, 100, 3)*128   # Use random data as an example
    lic_texture_data = lic_texture_data.astype('uint8')
    renderable_object = Renderable(lic_texture_data,5,2,-5)
    shaderManager=getShaderManager()
    shaderManager.add_shader_program("simpleColor","assets/shaders/simple_vertex.glsl","assets/shaders/simple_fragment.glsl")
    shaderManager.add_shader_program("colormapMat","assets/shaders/simple_vertex.glsl","assets/shaders/colorMap_fragment.glsl")
    defaultMat=shaderManager.getDefautlMaterial()
   
    # renderParamterPage=LicParameter("LicParameter")
    camera = Camera(60.0, (0, 0, 5), (0, 0, 0), [0.0, 1.0, 0.0],size[0],size[1])
    coord=CoordinateSystem(engine.scene)
    planarManifold=PlanarManifold(32,32)
    actFieldWidget=ActiveField()
    VectorGlyph=VertexArrayVectorGlyph()
    VectorGlyph.setMaterial(defaultMat)
    # nlpc=  NLPCommandObject()
    commandBar=MainUICommand("mainCommandUI")
    test=GuiTest()
    netCDF=NetCDFLoaderOBJ()
    
   #! todo: implement the following actions to the command bar
    
    # commandBar.addAction("optc training", lambda obj: load_vector_field(Engine.scene,"autosave") )
    # commandBar.addAction("save vector field", lambda obj: save_vector_field(Engine.scene,".autosave/"+actFieldWidget.getActiveFieldName()+".json") )
    # commandBar.addAction("load vector field", lambda obj: load_vector_field(Engine.scene,"autosave") )

    engine.addObjects2Scene([coord,planarManifold, actFieldWidget,commandBar,camera,VectorGlyph,netCDF]   )
    engine.eventRegister.register(lambda event: camera.eventCallBacks(event))
    engine.eventRegister.register(lambda event: actFieldWidget.eventCallBacks(event))
    engine.eventRegister.register(lambda event: engine.scene.save_state_all() if event.type == pygame.KEYDOWN and event.key == pygame.K_F3 else None)
    # eventRegister.register(lambda event: renderable_object.eventCallBacks(event))



    # args=config['training']
    # # device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # device=torch.device("cpu")
    # args['device'] = device    
    # args["epochs"]=50
    
    vectorField2d= rotation_four_center((32,32),32)
    actFieldWidget.insertField("rfc",vectorField2d)
    # resUfield=ObserverFieldOptimization(vectorField2d,args)
    # actFieldWidget.insertField("Result field",resUfield)
    # circle=VertexArrayObject("Cone")
    # circle.appendConeWithoutCommit(np.array([0,-1,0],dtype=np.float32),np.array([0,1,0],dtype=np.float32), 0.5, 2, 32)
    # circle.commit()
    
    # engine.scene.add_object(nlpc)
    # plane.appendArrowWithoutCommit(np.array([0,0,0],dtype=np.float32),np.array([1,0,0],dtype=np.float32),0.05,1.0, 0.2, 0.1, 8)
    # plane.commit()
    # plane.setGuiVisibility(False)
    # vertices, indices, textures= createPlane([32,32],[-2,-2,2,2])
    # plane.appendVertexGeometry(vertices, indices, textures)
    # plane.setMaterial(defaultMat)
    # Engine.scene.add_oject(plane)

    # plane2=VertexArrayObject("plane2")
    
    # plane2.appendVertexGeometry(vertices, indices, textures)
    # plane2.setMaterial(defaultMat)
    # scene.add_object(plane2)
    # v,t,i=create_cube()
    # cube=VertexArrayObject("cube")
    # cube.appendVertexGeometry(v, i,t)
    # cube.setMaterial(defaultMat)
    # scene.add_object(cube)

    engine.MainLoop()
    

if __name__ == "__main__":
    main()