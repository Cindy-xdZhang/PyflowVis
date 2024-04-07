import imgui
from .Object import Object,singleton
import logging

imguiStackMargin = 10
leftWindowsWidth = 305
# lastWindowHeightPolyscope = 200
# lastWindowHeightUser = 200
# rightWindowsWidth = 500

@singleton
class LoggingWidget(Object):
    def __init__(self, name):
        super().__init__(name)
        self.loggerOptionSet =  ["wandb","logging"]
        self.loggingLevelSet=  ["DEBUG","INFO","WARNING","ERROR","CRITICAL"]
        self.create_variable_gui("logger", self.loggerOptionSet, False)
        self.create_variable_gui("loggingLevel",    self.loggingLevelSet, False)
        self.updateOptionValue("logger", "logging")
        self.updateOptionValue("loggingLevel", "DEBUG")

    def updateOptionValue(self,key,value):
        super().updateOptionValue(key,value)
        if key=="loggingLevel":            
            if value=="DEBUG":
                logging.getLogger().setLevel(level=logging.DEBUG)
            elif value=="INFO":
                logging.getLogger().setLevel(level=logging.INFO)
            elif value=="WARNING":
                logging.getLogger().setLevel(level=logging.WARNING)
            elif value=="ERROR":
                logging.getLogger().setLevel(level=logging.ERROR)
            elif value=="CRITICAL":
                logging.getLogger().setLevel(level=logging.CRITICAL)


    def debug(self, *args):    
        logging.debug(*args)
    
    def info(self, *args):    
        logging.info(*args)

    def warning(self, *args):    
        logging.warning(*args)    
    def error(self, *args):    
        logging.error(*args)    

    def critical(self, *args):    
        logging.critical(*args)
        
         
def getlogger() -> LoggingWidget:
    return LoggingWidget("Logger")

class MainUICommand(Object):
   
    def __init__(self, name):
        super().__init__(name)
        self.screenshotExtension = ".png"
        self.screenshotTransparency = False
        self.posX,self.posY=None,None
        self.loggingWidget=getlogger()
            
        
        # self.pick=None      
          
    # def build_pick_gui(self):
    #     global rightWindowsWidth, lastWindowHeightUser  # Assuming these are managed globally
    #     if self.pick is not None:#pick a structure in the scene
    #         imgui.set_next_window_position(view.windowWidth - (rightWindowsWidth + imguiStackMargin),
    #                                     2 * imguiStackMargin + lastWindowHeightUser)
    #         imgui.set_next_window_size(rightWindowsWidth, 0.)

    #         imgui.begin("Selection", None)
    #         selection = pick.get_selection()

    #         imgui.text_unformatted(f"{selection.first.type_name()}: {selection.first.name}")
    #         imgui.separator()
    #         selection.first.build_pick_ui(selection.second)

    #         rightWindowsWidth = imgui.get_window_width()
    #         imgui.end()
            

    def drawGui(self):
        show_polyscope_window = True
        _, show_polyscope_window = imgui.begin(self.name,show_polyscope_window)
        imgui.set_next_window_size(leftWindowsWidth, 0.)
        self.posX,self.posY= imgui.get_item_rect_min()
        if imgui.button("Reset View"):
            camera = self.parentScene.getObject("Camera")
            if camera is not None:
                camera.resetCamera()
        imgui.same_line()
        
        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (1.0, 0.0))
        if imgui.button("Screenshot"):
            print("buton Screenshot pressed, no callback  yet  ")
            
        imgui.same_line()
        if imgui.arrow_button("##Option", imgui.DIRECTION_DOWN):
            imgui.open_popup("ScreenshotOptionsPopup")
        imgui.pop_style_var()
        
        if imgui.begin_popup("ScreenshotOptionsPopup"):
            _, self.screenshotTransparency = imgui.checkbox("with transparency", self.screenshotTransparency)
            if imgui.begin_menu("file format"):
                clicked_png, _ = imgui.menu_item(".png", None, self.screenshotExtension == ".png")
                if clicked_png:
                    self.screenshotExtension = ".png"
                clicked_jpg, _ = imgui.menu_item(".jpg", None, self.screenshotExtension == ".jpg")
                if clicked_jpg:
                    self.screenshotExtension = ".jpg"
                imgui.end_menu()

            imgui.end_popup()

        imgui.same_line()
        if imgui.button("Controls"):
            pass  # Placeholder for hover state logic
        if imgui.is_item_hovered():
            
            imgui.set_next_window_position(imguiStackMargin +self.posX+leftWindowsWidth, self.posY)
            imgui.set_next_window_size(0., 0.)
            imgui.begin("Controls", None, imgui.WINDOW_NO_TITLE_BAR)
            imgui.text_unformatted("View Navigation:")
            imgui.text_unformatted("      Rotate: [left click drag]")
            imgui.text_unformatted("   Translate: [shift] + [left click drag] OR [right click drag]")
            imgui.text_unformatted("        Zoom: [scroll] OR [ctrl] + [shift] + [left click drag]")
            imgui.text_unformatted("   Use [ctrl-c] and [ctrl-v] to save and restore camera poses")
            imgui.text_unformatted("     via the clipboard.")
            imgui.text_unformatted("\nMenu Navigation:")
            imgui.text_unformatted("   Menu headers with a '>' can be clicked to collapse and expand.")
            imgui.text_unformatted("   Use [ctrl] + [left click] to manually enter any numeric value")
            imgui.text_unformatted("     via the keyboard.")
            imgui.text_unformatted("   Press [space] to dismiss popup dialogs.")
            imgui.text_unformatted("\nSelection:")
            imgui.text_unformatted("   Select elements of a structure with [left click]. Data from")
            imgui.text_unformatted("     that element will be shown on the right. Use [right click]")
            imgui.text_unformatted("     to clear the selection.")
            imgui.end()
   
        #   draw logger
        if imgui.tree_node("logger", imgui.TREE_NODE_DEFAULT_OPEN):
            self.loggingWidget.DrawPropertiesInGui(self.loggingWidget.persistentProperties)
            self.loggingWidget.DrawPropertiesInGui(self.loggingWidget.nonPersistentProperties)
            imgui.tree_pop()


        self.DrawPropertiesInGui(self.persistentProperties)
        self.DrawPropertiesInGui(self.nonPersistentProperties)
        self.DrawActionButtons()

        #   fps
        io = imgui.get_io()
        fps = io.framerate
        ms_per_frame = 1000.0 / fps if fps != 0 else 0.0
        imgui.text(f"{ms_per_frame:.1f} ms/frame ({fps:.1f} FPS)")
        imgui.end()
