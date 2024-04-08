import pygame
import OpenGL.GL as gl
from OpenGL.GL import *
from OpenGL.GLU import *
import imgui
from imgui.integrations.pygame import PygameRenderer
import numpy as np
import logging
from GuiObjcts  import *
from EventRegistrar import EventRegistrar
from GuiObjcts.Object import Scene, singleton

def getEngine():
    return VisualizationEngine({})

@singleton
class VisualizationEngine:
    def __init__(self,config) -> None:
        size=config["window_size"]
        pygame.init()
        # pygame.font.init()
        # font_size = 50
        # self.font = pygame.font.SysFont("Arial", font_size)
        self.screen=pygame.display.set_mode(size,  pygame.DOUBLEBUF | pygame.OPENGL| pygame.RESIZABLE|pygame.HWSURFACE)
    
         # Set up OpenGL context
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_TEXTURE_2D)  # Enable texture mapping
        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glViewport(0, 0, size[0], size[1])
        self.scene=Scene("DefaultScene")
        # Setup ImGui context and the pygame renderer for ImGui
        imgui.create_context()
        imgui.get_io().display_size = size
        imgui.get_io().fonts.get_tex_data_as_rgba32()
        self.impl=PygameRenderer()
        self.eventRegister=EventRegistrar(self.impl)
        self.initLoggingSetting()
        
    def initLoggingSetting(self):
        class CustomFormatter(logging.Formatter):
            grey = "\x1b[38;21m"
            yellow = "\x1b[33;21m"
            red = "\x1b[31;21m"
            bold_red = "\x1b[31;1m"
            reset = "\x1b[0m"
            format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

            FORMATS = {
                logging.DEBUG: grey + format + reset,
                logging.INFO: grey + format + reset,
                logging.WARNING: yellow + format + reset,
                logging.ERROR: red + format + reset,
                logging.CRITICAL: bold_red + format + reset
            }
            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno)
                formatter = logging.Formatter(log_fmt)
                return formatter.format(record)

        ch = logging.StreamHandler()
        ch.setFormatter(CustomFormatter())
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(ch)

    def setUpSceneObjects(self,ObjectList=None):
        # all the objects in the scene
        for obj in ObjectList:
            self.scene.add_object(obj)
        self.scene.restore_state_all()

    def getScene(self) -> Scene:
        return self.scene
    

    def MainLoop(self):
        # clock = pygame.time.Clock()
        while self.eventRegister.running:
            self.eventRegister.handle_events()
            # Rendering
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
            self.scene.draw_all()

            imgui.new_frame()
            self.scene.drawGui()
            imgui.render()
            imgui.end_frame()
            self.impl.render(imgui.get_draw_data())

            pygame.display.flip()
        
        # scene.save_state_all()
        self.impl.shutdown()
        pygame.quit()
        