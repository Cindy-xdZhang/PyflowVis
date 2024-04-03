from   .Object import *
import pygame

class LICRender(Object):
    def __init__(self,name):
        super().__init__(name)
        self.create_variable("StepSize",0.01,True,0.01)
        self.create_variable("MaximumStepSize",100000,True,1) 
        self.create_variable("NoiseImage","assets//noise//512x512.png",True)    
    def render(self,shader):
        # Get the noise image
        noise_image = self.load_image(self.NoiseImage)
        
        # Get the vector field
        vx, vy = self.get_vector_field()
        
        # Apply LIC
        lic_image = self.apply_lic(noise_image, vx, vy)
        
        # Render the LIC image
        self.render_lic(lic_image)
  
class ActiveField(Object):
    def __init__(self):
        super().__init__("ActiveField")
        self.pause=False
        self.create_variable_gui("time",0.0,False, {'widget': 'input'})
        self.create_variable_gui("animationSpeed",0.01,False, {'widget': 'input'})
        #list of str is treated specially  as option in my gui implementation, don't need to specify customization, it always render as combo box
        self.create_variable_gui("active field",[""],False)
        self.activeField= {}


    def time(self)->float:
        return self.getValue("time")
    
    def draw(self):
        time=self.getValue("time")
        if self.pause==False and  0<=time<2*np.pi:#running the animation
            time+=self.getValue("animationSpeed")
            self.setValue("time",time)
        elif self.pause==False:
            self.pause=True

    def eventCallBacks(self,event):        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            self.pause = not self.pause
            time=self.getValue("time")
            time=0.0 if time>= 2*np.pi and self.pause==False else time
            self.setValue("time",time)

    def insertField(self,fieldName:str,field):
        if field is None or fieldName is None:
            return
        self.activeField[fieldName]=field
        fieldNameList=self.getValue("active field")
        if fieldName not in fieldNameList:
            fieldNameList.append(fieldName)
            self.setValue("active field",fieldNameList)

    def getActiveFieldName(self)->str:
        return self.getOptionValue("active field")
    def getActiveField(self):
        if self.getActiveFieldName() in self.activeField:
            return self.activeField[self.getActiveFieldName()]
        else:
            return None
    
    def getField(self,fieldName:str):
        return self.activeField[fieldName]


