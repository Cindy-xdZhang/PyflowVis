from   .Object import *
from  FLowUtils.VectorField2d import *
from FLowUtils.VectorField3d import *
from typeguard import typechecked
from FLowUtils.ScalarField2d import *
from .VisualizationEngine import getEngine
from FLowUtils.FTLE import *
from FLowUtils.ScalarField2d import _format_float
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
  
class ActiveFieldObj(Object):
    def __init__(self):
        super().__init__("ActiveField")
        self.pause=True

        def timedirtyCallBack(obj:ActiveFieldObj) -> None:
            vectorGlyph=obj.parentScene.getObject("vectorGlyph")
            if vectorGlyph is not None:
                vectorGlyph.dirty=True
            flowlineOBj=obj.parentScene.getObject("Flowline")
          
            if flowlineOBj is not None and flowlineOBj.getValue("streamline_active"):
                flowlineOBj.streamline_dirty=True

        def updateActivefieldcb(obj:ActiveFieldObj) -> None:
            timedirtyCallBack(obj)
            activeField=obj.getActiveField()
            if activeField is not None:
                animationSpeed=0.5*activeField.timeInterval
                obj.updateValue("animationSpeed",animationSpeed)
                getEngine().eventRegister.notifyEvent("activefield_changed")
        
        def updateActiveScalarfieldcb(obj:ActiveFieldObj) -> None:
           planarObj=obj.parentScene.getObject("plane")
           planarObj.setScalarField(obj.getActiveScalarField())

    

        self.create_variable_callback_with_gui_customization("time",0.0,timedirtyCallBack,False, {'widget': 'input'})
        self.create_variable_gui("animationSpeed",0.01,False, {'widget': 'input'})
        #list of str is treated specially  as option in my gui implementation, don't need to specify customization, it always render as combo box
        self.create_variable_callback_with_gui_customization("active field",[],updateActivefieldcb,False)
        self.create_variable_gui("observer field",[],False)


        self.scalarFieldManager=ScalarFieldManager()
        self.create_variable_callback_with_gui_customization("active scalar field",[],updateActiveScalarfieldcb,False)


        self.create_variable_gui("scalarFieldOperation", ["MAGNITUDE","CURL","Q_CRITERION","LAMBDA2","IVD"], False)
        self.addAction("compute scalar field",lambda obj:obj.requestScalarField())

        self.create_variable_gui("scalarfield_path", "", False, {'widget': 'file_dialog'})
        self.create_variable("FTLE_dt_MaxIter_upSampling_timeUps", np.array([0.01,2000,2,1]) ,True)

        def FTLE_2D_CUDA_action(*args, **kwargs):
            actFieldWidget:ActiveFieldObj = self
            vector_field:UnsteadyVectorField2D = actFieldWidget.getActiveField()
            activeFieldName=actFieldWidget.getActiveFieldName()
            if vector_field is None:
                return
            ftle_params=self.getValue("FTLE_dt_MaxIter_upSampling_timeUps")
            FTLE_dt,MaxIter,upSampling,timeUps=ftle_params[0],ftle_params[1],ftle_params[2],ftle_params[3]
            resultScalarFieldSlice=compute_FTLE_2D_field_CUDA(vector_field, FTLE_dt, MaxIter,upSampling,timeUps)
            FTLE_param_str=f"_dt{_format_float(FTLE_dt)}_MaxIter{MaxIter}_upSampling{_format_float(upSampling)}_timeUps{_format_float(timeUps)}"

            actFieldWidget.insertScalarField(activeFieldName +FTLE_param_str,resultScalarFieldSlice)     

            ridge_mask = extract_ridges_2d_amr_mask(resultScalarFieldSlice,  
            max_level=3,          # 细化到 2^3=8 倍 finest 粗起点
            neighbor_range=1,     # ridge 周围 1 环细化带
            s_min=None,           # 或者设一个 FTLE 阈值，比如 0.2
            lambda_max=None,      # 或设更负的阈值，如 -1e-3
            lookahead_cells=0,    # 若要更激进可>0（此处已用更强检测，通常用不上）
            lookahead_by='height')

            # get meta information from ridge mask
            RidgeMask_grid_size=ridge_mask.shape#shape is (T,Y,X)
            RidgeMask_domainMinBoundary=vector_field.domainMinBoundary
            RidgeMask_domainMaxBoundary=vector_field.domainMaxBoundary

            ridge_maskField=ScalarField2D(RidgeMask_grid_size[2], RidgeMask_grid_size[1], RidgeMask_grid_size[0], RidgeMask_domainMinBoundary, RidgeMask_domainMaxBoundary)
            ridge_maskField.set_discrete_data(ridge_mask)
            actFieldWidget.insertScalarField(activeFieldName +FTLE_param_str+"_ridge",ridge_maskField)



        self.addAction("FTLE_2D_CUDA", FTLE_2D_CUDA_action)

        def saveActiveFieldAction(*args, **kwargs) -> None:
            actFieldWidget:ActiveFieldObj = self
            activeField=actFieldWidget.getActiveScalarField()
            activeFieldName=actFieldWidget.getActiveScalarFieldName()
            if activeField is not None:
                self.scalarFieldManager.save_scalar_field_to_file(activeField, "outputs/scalarFields",activeFieldName)

        self.addAction("save active scalarfield",saveActiveFieldAction)
        

        def loadActiveFieldAction(*args, **kwargs) -> None:
            resolved_path = self.getValue("scalarfield_path")
            if  not isinstance(resolved_path, str) or resolved_path.strip() == "":
                import tkinter as tk
                from tkinter import filedialog
                try:
                    root = tk.Tk()
                    root.withdraw()  # Hide the main window
                    file_path = filedialog.askopenfilename()
                    if file_path:  # Only update if a file was selected
                        self.setValue("scalarfield_path",file_path)
                        resolved_path=file_path
                except Exception as e:
                    print(f"Filedialog Error: {e}")
                    return
            loaded_scalar_field,loaded_scalar_fieldName=self.scalarFieldManager.load_scalar_field_from_file(resolved_path)
            if loaded_scalar_field is not None:
                self.insertScalarField(loaded_scalar_fieldName,loaded_scalar_field)

        self.addAction("load scalar field",loadActiveFieldAction)

        # ── Reference-frame optimization (killing / Günther17) ────────────────
        # 对当前 active field 做参考系优化，把观察者场 u 与观测场 v-u 插回可视化。
        # 详见 docs/referenceframeOptimization.md, FLowUtils/ReferenceFrameOptimization.py
        self.create_variable("RFO_neighborhood", 4, True)  # Günther 逐点邻域半径 U

        def RFO_action(mode: str):
            actFieldWidget: ActiveFieldObj = self
            vector_field = actFieldWidget.getActiveField()
            name = actFieldWidget.getActiveFieldName()
            if vector_field is None:
                logging.getLogger().warning("RFO: no active field selected")
                return
            from FLowUtils.ReferenceFrameOptimization import (
                killing_optimization_2d, killing_optimization_3d,
                gunther17_optimization_2d, gunther17_optimization_3d,
            )
            is3d = vector_field.getDim() == 3
            nb = int(self.getValue("RFO_neighborhood"))
            if mode == "killing":
                res = (killing_optimization_3d if is3d else killing_optimization_2d)(vector_field)
                if res.params is not None and len(res.params) > 2:
                    logging.getLogger().info(
                        f"RFO killing '{name}': params mean over t = {np.round(res.params[1:-1].mean(0), 4)}")
            else:
                res = (gunther17_optimization_3d if is3d else gunther17_optimization_2d)(vector_field, neighborhood=nb)
            actFieldWidget.insertField(f"{name}_{mode}_u", res.u_field)            # 观察者场 u
            actFieldWidget.insertField(f"{name}_{mode}_v-u", res.v_minus_u_field)  # 观测场 v-u
            logging.getLogger().info(f"RFO {mode} '{name}': inserted '{name}_{mode}_u' and '{name}_{mode}_v-u'")

        self.addAction("RFO killing (observer u + v-u)", lambda *a, **k: RFO_action("killing"))
        self.addAction("RFO gunther17 (observer u + v-u)", lambda *a, **k: RFO_action("gunther"))

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
        if self.pause==False and self.getActiveField().getMinTime()<=time<self.getActiveField().getMaxTime():#running the animation
            time+=self.getValue("animationSpeed")
            self.setValue("time",time)
        elif self.pause==False:
            self.pause=True

    def eventCallBacks(self,event):       
        import pygame
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            self.pause = not self.pause
            time=self.getValue("time")
            time= self.getActiveField().getMinTime()if time>= self.getActiveField().getMaxTime() and self.pause==False else time
            self.setValue("time",time)

    @typechecked
    def insertField(self,fieldName:str,field:UnsteadyVectorField2D|UnsteadyVectorField3D|None):
        if field is None or fieldName is None:
            return
        self.activeField[fieldName]=field
        fieldNameList=self.getValue("active field")
        if fieldName not in fieldNameList:
            fieldNameList.append(fieldName)
            self.setValue("active field",fieldNameList,False)
        
        self.setValue("observer field",fieldNameList,False)
        # if  only one field exist,  make it active
        if len(fieldNameList)==1:
            self.updateOptionValue("active field",fieldNameList[0])
            self.updateOptionValue("observer field",fieldNameList[0])

    def getActiveObserverField(self):
        if self.getActiveObserverFieldName() in self.activeField:
            return self.activeField[self.getActiveObserverFieldName()]
        else:
            return None

    def getActiveObserverFieldName(self):
        return self.getOptionValue("observer field")

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
