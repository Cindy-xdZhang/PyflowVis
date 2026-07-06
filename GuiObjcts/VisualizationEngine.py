import pygame
import OpenGL.GL as gl
from OpenGL.GL import *
from OpenGL.GLU import *
import imgui
from imgui.integrations.pygame import PygameRenderer
import numpy as np
import logging
from GuiObjcts  import *
from GuiObjcts.EventRegistrar import EventRegistrar
from GuiObjcts.Object import Scene, singleton,getScene
from GuiObjcts.shaderManager import *
from GuiObjcts.CameraObject import *
from DeepUtils.MiscFunctions import initLogging


def getEngine():
    return VisualizationEngine({})

@singleton
class VisualizationEngine:
    def __init__(self,config) -> None:
        self.config=config
        size=config["window_size"]
        pygame.init()
        self.screen=pygame.display.set_mode(size,  pygame.DOUBLEBUF | pygame.OPENGL| pygame.RESIZABLE|pygame.HWSURFACE)
         # Set up OpenGL context
        version = gl.glGetString(gl.GL_VERSION)
        print("OpenGL version:", version.decode())
        print("GLU version:",gluGetString(GLU_VERSION))

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_TEXTURE_2D)  # Enable texture mapping
        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glViewport(0, 0, size[0], size[1])
        self.scene=getScene()
        # Setup ImGui context and the pygame renderer for ImGui
        imgui.create_context()
        imgui.get_io().display_size = size
        imgui.get_io().fonts.get_tex_data_as_rgba32()
        self.impl=PygameRenderer()
        self.eventRegister=EventRegistrar(self.impl)
        self.camera=None        
        initLogging()

    def initFront(self):
        pass
        # pygame.font.init()
        # font_size = 50
        # self.font = pygame.font.SysFont("Arial", font_size)

 
    def getBuiltInTextureNames(self):
        return getTextureManager().getBuiltInTextureNames()
    
    def addObjects2Scene(self,ObjectList=None):
        # all the objects in the scene
        for obj in ObjectList:
            self.scene.add_object(obj)
  

    def getScene(self) -> Scene:
        return self.scene
    

    def finalizeSettleUp(self):
        """
        Call this function after all the objects are added to the scene and before engine main loop.
        This function will restore objects, fixed down camera, and assign the camera to the objects.
        """
        
        self.scene.postInit()
        self.camera=self.scene.getObject("Camera")
        if self.config['restore']:
            self.scene.restore_state_all()

    
    


    def MainLoop(self):
        # clock = pygame.time.Clock()
        while self.eventRegister.running:
            self.eventRegister.handle_events()
            # Rendering
            gl.glClearColor(0.1, 0.1, 0.1, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT|  gl.GL_STENCIL_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST);
        
            self.scene.render_all()
            gl.glUseProgram(0)
            gl.glBindVertexArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            # imgui's fixed-pipeline renderer samples its font atlas from texture unit 0 as a
            # GL_TEXTURE_2D. Any object that bound a sampler texture (DVR's 3D volume, the 1D-array
            # colormap, ...) leaves unit 0 on a non-2D target with the active unit off 0, which
            # blanks the ENTIRE GUI. Reset the active unit + unbind those targets before imgui.
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
            gl.glBindTexture(gl.GL_TEXTURE_1D_ARRAY, 0)


            imgui.new_frame()
            self.scene.drawGui()
            imgui.render()
            imgui.end_frame()
            self.impl.render(imgui.get_draw_data())

            pygame.display.flip()
        
        # scene.save_state_all()
        self.impl.shutdown()
        pygame.quit()
        







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
            from GuiObjcts.IsoSurfaceObject import IsoSurfaceObject
            return IsoSurfaceObject()
        elif ObjectName.lower()=="VolumeRender".lower():
            from GuiObjcts.VolumeRenderObject import VolumeRenderObject
            return VolumeRenderObject()

    if packageName=="Basic2DFlow":
        ObjectsNameList=["CoordinateSystem","PlanarManifold","ActiveField","VectorGlyph","Indicator","NetCDFLoader","Flowline","IsoSurface","VolumeRender"]
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



