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

class VisualizationEngine:
    def __init__(self,config) -> None:
        size=config["window_size"]
        pygame.init()
        pygame.display.set_mode(size,  pygame.DOUBLEBUF | pygame.OPENGL| pygame.RESIZABLE|pygame.HWSURFACE)
        # Configure logging to display all messages
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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

    def setUpSceneObjects(self,ObjectList=None):
        # all the objects in the scene
        for obj in ObjectList:
            self.scene.add_object(obj)
        self.scene.restore_state_all()

    def getScene(self):
        return self.scene
    

    def MainLoop(self):
        # clock = pygame.time.Clock()
        while self.eventRegister.running:
            self.eventRegister.handle_events()
            # Rendering
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        
            imgui.new_frame()
            self.scene.drawGui()
            imgui.render()
            imgui.end_frame()
            self.impl.render(imgui.get_draw_data())
            self.scene.draw_all()
    
            pygame.display.flip()
            # pygame.time.wait(10)# Limit to 60 frames per second
        
        # scene.save_state_all()
        self.impl.shutdown()
        pygame.quit()
        