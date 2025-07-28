from   .Object import *
from  FLowUtils.VectorField2d import *
from FLowUtils.VectorField3d import *
import pygame
from typeguard import typechecked
from FLowUtils.ScalarField2d import *
from .VisualizationEngine import getEngine

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
        self.pause=True

        def dirtyCallBack(obj:ActiveField) -> None:
            vectorGlyph=obj.parentScene.getObject("vectorGlyph")
            if vectorGlyph is not None:
                vectorGlyph.dirty=True
            flowlineOBj=obj.parentScene.getObject("flowline")
            if flowlineOBj is not None and flowlineOBj.getValue("pathline_active"):
                flowlineOBj.pathline_dirty=True
            if flowlineOBj is not None and flowlineOBj.getValue("streamline_active"):
                flowlineOBj.streamline_dirty=True

        def updateActivefieldcb(obj:ActiveField) -> None:
            dirtyCallBack(obj)
            activeField=obj.getActiveField()
            if activeField is not None:
                animationSpeed=0.5*activeField.timeInterval
                obj.updateValue("animationSpeed",animationSpeed)
                getEngine().eventRegister.notifyEvent("activefield_changed")
        
        def updateActiveScalarfieldcb(obj:ActiveField) -> None:
           planarObj=obj.parentScene.getObject("plane")
           planarObj.setScalarField(obj.getActiveScalarField())


        self.create_variable_callback_with_gui_customization("time",0.0,dirtyCallBack,False, {'widget': 'input'})
        self.create_variable_gui("animationSpeed",0.01,False, {'widget': 'input'})
        #list of str is treated specially  as option in my gui implementation, don't need to specify customization, it always render as combo box
        self.create_variable_callback_with_gui_customization("active field",[],updateActivefieldcb,False)

        self.scalarFieldManager=ScalarFieldManager()
        self.create_variable_callback_with_gui_customization("active scalar field",[],updateActiveScalarfieldcb,False)
        self.create_variable_gui("scalarFieldOperation", ["MAGNITUDE","CURL","Q_CRITERION","LAMBDA2","IVD"], False)
        self.addAction("compute scalar field",lambda obj:obj.requestScalarField())


        self.activeField= {}

    def requestScalarField(self):
        operation=self.getOptionValue("scalarFieldOperation")
        targetFieldName=self.getActiveFieldName()
        targetField=self.getField(targetFieldName)
        if targetField is None:
            logging.getLogger().warning(f"Field {targetFieldName} not found")
            return
        scalarField,resultName=self.scalarFieldManager.request_scalar_field(targetFieldName,targetField,operation)
        self.insertScalarField(resultName,scalarField)


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

    @typechecked
    def insertField(self,fieldName:str,field:UnsteadyVectorField2D|UnsteadyVectorField3D):
        if field is None or fieldName is None:
            return
        self.activeField[fieldName]=field
        fieldNameList=self.getValue("active field")
        if fieldName not in fieldNameList:
            fieldNameList.append(fieldName)
            self.setValue("active field",fieldNameList,False)
        # if  only one field exist,  make it active
        if len(fieldNameList)==1:
            self.updateOptionValue("active field",fieldNameList[0])

    def getActiveFieldName(self)->str:
        return self.getOptionValue("active field")
    def getActiveField(self):
        if self.getActiveFieldName() in self.activeField:
            return self.activeField[self.getActiveFieldName()]
        else:
            return None
    
    def getField(self,fieldName:str):
        return self.activeField[fieldName]
    
    def getActiveScalarFieldName(self):
        return self.getOptionValue("active scalar field")
    
    def getActiveScalarField(self):
        if self.getActiveScalarFieldName() in self.activeField:
            return self.activeField[self.getActiveScalarFieldName()]
        else:
            return None

    @typechecked
    def insertScalarField(self,fieldName:str,scalarField:ScalarField2D):
        if scalarField is None or fieldName is None:
            return
        self.activeField[fieldName]=scalarField
        fieldNameList=self.getValue("active scalar field")
        if fieldName not in fieldNameList:
            fieldNameList.append(fieldName)
            self.setValue("active scalar field",fieldNameList,False)
        # if  only one field exist,  make it active 
        if len(fieldNameList)==1:
            self.updateOptionValue("active scalar field",fieldNameList[0])
