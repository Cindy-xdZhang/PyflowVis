from GuiObjcts.VisualizationEngine import *
from FLowUtils.AnalyticalFlowCreator import *
from DeepUtils.utils import EasyConfig
from GuiObjcts.VertexArrayObject import *
from GuiObjcts.vectorGlyphObject import *
from GuiObjcts.ObjectGUIReflection import ValueGuiCustomization
from GuiObjcts.shaderManager import *
from FLowUtils.VectorField2d import *
from  GuiObjcts.PlanarManifold import *
from misc.fileMonitor import *
from GuiObjcts.netCDFObject import *
from GuiObjcts.FlowLineRenderObject import *
from GuiObjcts.IsoSurfaceObject import *
from GuiObjcts.VolumeRenderObject import *
from GuiObjcts.Indicator import *
from GuiObjcts.geometryRender import *
from GuiObjcts.ActiveFieldObject import *
from FLowUtils.flowDatasetUtils.JHTDB_Lodader import JHTDB_Lodader
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader


from typing import Tuple
def test_opengl_state():
    program = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
    depth_test = gl.glIsEnabled(gl.GL_DEPTH_TEST)
    cull_face = gl.glIsEnabled(gl.GL_CULL_FACE)
    blend = gl.glIsEnabled(gl.GL_BLEND)
    framebuffer = gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)
    viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
    polygon_mode = gl.glGetIntegerv(gl.GL_POLYGON_MODE)
    offset_factor = gl.glGetIntegerv(gl.GL_POLYGON_OFFSET_FACTOR)
    offset_units = gl.glGetIntegerv(gl.GL_POLYGON_OFFSET_UNITS)
    offset_factor2 = gl.glGetIntegerv(gl.GL_POLYGON_OFFSET_FACTOR)
    print(f"Current OpenGL state:")
    print(f"Program: {program}")
    print(f"Depth test enabled: {depth_test}")
    print(f"Face culling enabled: {cull_face}")
    print(f"Blending enabled: {blend}")
    print(f"Framebuffer binding: {framebuffer}")
    print(f"Viewport: {viewport}")
    print(f"Polygon mode: {polygon_mode}")
    print(f"Polygon offset factor: {offset_factor}")
    print(f"Polygon offset units: {offset_units}")
    print(f"Polygon offset factor (2): {offset_factor2}")

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
   
   


def buildWorkLoads(packageName:str):
    def __buld_an_object(ObjectName:str):
        if ObjectName.lower()=="CoordinateSystem".lower():
            return CoordinateSystem()
        elif ObjectName.lower()=="PlanarManifold".lower():
            return PlanarManifold()
        elif ObjectName.lower()=="ActiveField".lower():
            return ActiveFieldObj()
        elif ObjectName.lower()=="VectorGlyph".lower():
            return VertexArrayVectorGlyph()
        elif ObjectName.lower()=="Indicator".lower():
            return Indicator()
        elif ObjectName.lower()=="NetCDFLoader".lower():
            return NetCDFLoaderOBJ()
        elif ObjectName.lower()=="Flowline".lower():
            return FlowLineObject()
        elif ObjectName.lower()=="IsoSurface".lower():
            return IsoSurfaceObject()
        elif ObjectName.lower()=="VolumeRender".lower():
            return VolumeRenderObject()
        elif ObjectName.lower()=="GeneralMeshRenderer".lower():
            from GuiObjcts.GeneralMeshRenderObject import GeneralMeshRenderObject
            return GeneralMeshRenderObject()

    if packageName=="Basic2DFlow":
        # NOTE: VolumeRender (DVR) is kept LAST so it composites over all opaque geometry and can
        # clamp its rays to the finished depth buffer (depth-aware compositing).
        ObjectsNameList=["CoordinateSystem","PlanarManifold","ActiveField","VectorGlyph","Indicator","NetCDFLoader","Flowline","IsoSurface","GeneralMeshRenderer","VolumeRender"]
    elif packageName=="Basic3DFlow":
        pass
    else:
        raise ValueError(f"Invalid package name: {packageName}")

    returnDict={}
    for ObjectName in ObjectsNameList:
        obj=__buld_an_object(ObjectName)
        returnDict[ObjectName]=obj
    return returnDict


def init_render(cfg="config/renderingConfig.yaml"):
    config = EasyConfig()
    config.load(cfg, recursive=False)
    engine=VisualizationEngine(config=config['rendering'])
    size=config['rendering']["window_size"]
    camera = Camera(60.0, (0, 0, 10), (0, 0, 0), [0.0, 1.0, 0.0],size[0],size[1])
    engine.addObjects2Scene([camera])
    
    ObjectNameDict=buildWorkLoads("Basic2DFlow")
    ObjectList=ObjectNameDict.values()
    engine.addObjects2Scene(ObjectList)
    engine.finalizeSettleUp()

    #built-in events
    engine.eventRegister.register(lambda event: engine.scene.save_state_all() if event.type == pygame.KEYDOWN and event.key == pygame.K_F3 else None)
    engine.eventRegister.register(lambda event: camera.eventCallBacks(event))
    if engine.scene.hasObject("Indicator"):
        indicator=engine.scene.getObject("Indicator")
        engine.eventRegister.register(lambda event: indicator.eventCallBacks(event))
    else:
        logging.getLogger().fatal("No Indicator object found in scene")
    if engine.scene.hasObject("ActiveField"):
        actFieldWidget=engine.scene.getObject("ActiveField")        
    else:
        logging.getLogger().fatal("No ActiveField object found in scene")
    return engine,camera,ObjectNameDict






def main():

    engine,camera,ObjectNameDict=init_render()
    actFieldWidget=ObjectNameDict["ActiveField"]

    #####################################################################################
    #########################reference frame transformation test#########################
    #####################################################################################
    # vectorField2d= rotation_four_center((64,64),64)
    # actFieldWidget.insertField("rfc",vectorField2d)
    # vectorField2d2= constant_rotation((64,64),64)
    # actFieldWidget.insertField("constant_rotation",vectorField2d2)


    #####################################################################################
    #########################3d flow vis test#########################
    #####################################################################################
    flow_asset_folder="C:\\Users\\xingdi\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowData3D"
    cylider_netCDF=os.path.join(flow_asset_folder,"tornado3d.nc")
    vectorField3d= NetCDFLoader.load_vector_field3d(cylider_netCDF,64);
    actFieldWidget.insertField("tornado3d",vectorField3d)

    
    #exmaple loading amria file
    # vectorField2d=AmiraLoader.load_vector_field2d("C:\\Users\\xingdi\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowData2D\\GerrisFlowSolverData\\0190.am")
    # actFieldWidget.insertField("resampled",vectorField2d)
    # vectorField2d2=AmiraLoader.load_vector_field2d("C:\\Users\\xingdi\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowData2D\\GerrisFlowSolverDataTemp\\0071.am")
    # actFieldWidget.insertField("original",vectorField2d2)


    #jhtdb turlulent 2d flow vis load example test
    # jhtdb_loader=JHTDB_Lodader()
    # vf2d=jhtdb_loader.load_2d_unsteadyFlow("channel5200","xz",64,64, 0.000997615051193464,(0, 0.4*np.pi),(0, 0.15*np.pi),1,11,11,integerTime=True)
    # actFieldWidget.insertField("channel",vf2d)





    # args=config['training']
    # # device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # device=torch.device("cpu")
    # args['device'] = device    
    # args["epochs"]=50
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