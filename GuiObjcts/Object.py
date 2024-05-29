import jsonpickle
import unittest
import os
import imgui
from typing import Dict, Any
from typeguard import typechecked
import logging
from .ObjectGUIReflection import *
# from functools import lru_cache
import hashlib

def singleton(cls):
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    inner._original_class = cls  # Expose the original class
    return inner


def operate_on_dict(dict_to_operate, key_list, new_value, index=0):
        """
        Recursively navigates through a nested dictionary using a list of keys and updates the value at the
        deepest level specified by these keys.

        :param dict_to_operate: The dictionary to operate on.
        :param key_list: A list of keys specifying the path to the value to be updated.
        :param new_value: The new value to be set at the specified path.
        :param index: The current index in key_list being processed.
        # Example usage:
        # Commenting out the call to ensure compliance with instructions.
        #dict_example = {"key1": {"key2": {"key3": "original_value"}}}
        #key_list = ["key1", "key2", "key3"]
        #new_value = "updated_value"
        #operate_on_dict(dict_example, key_list, new_value)
        #print(dict_example)
        """
        # Check if we've reached the end of the key list
        if index == len(key_list) - 1:
            # Update the value
            dict_to_operate[key_list[index]] = new_value
            return
        
        key = key_list[index]
        # Check if the current key exists and leads to a dictionary
        if key in dict_to_operate and isinstance(dict_to_operate[key], dict):
            # Recurse with the next level of the dictionary and the next index
            operate_on_dict(dict_to_operate[key], key_list, new_value, index + 1)
        else:
            # Handle cases where the path does not lead to a dictionary
            raise KeyError(f"Key '{key}' does not exist or does not lead to a dictionary at index {index}.")  


 
        



    
class Object:
    def __init__(self, name:str,autoSaveFolderPath = "autosave"):
        self.name = name
        self.persistentPropertyDefaultValues={}
        self.persistentProperties = {}
        self.nonPersistentProperties = {}
        self.autoSaveFolderPath =autoSaveFolderPath
        self.customizations=[]
        self.actions = {}
        self.callbacks={}
        self.optionValues = {}
        self.GuiVisible=True
        self.renderVisible=False
        self.parentScene=None
        self.cameraObject=None

    def getParentScene(self):
        return self.parentScene
    
    def setCamera(self,camera):
        self.cameraObject=camera 

    def setUpScene(self,SceneInscane):
        self.parentScene=SceneInscane if isinstance(SceneInscane, Scene._original_class) else None

    def _get_hash_key(self):
        """this function hash the object properties;
        which is usd like to accerlate a slow function if it is called with the same properties
        # @lru_cache(maxsize=128, typed=True)
        def getScope(self ):
         .....

        """
        hash_str = (str(self.persistentProperties) +
                    str(self.nonPersistentProperties) +
                    str(self.optionValues))
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    # @lru_cache(maxsize=128, typed=True)
    def getScope(self ):
        AllVAriables=  {**self.persistentProperties, **self.nonPersistentProperties}
        #!todo for optionValues we can save current Selection as uint instead of "str"
        for optionName,optionValueStr in self.optionValues.items():
            OptionList=self.getValue(optionName)
            #count the index of optionValueStr in OptionList
            if OptionList is not None:
                index=OptionList.index(optionValueStr)
                AllVAriables[optionName]=int(index)
        
        return AllVAriables

    @typechecked
    def setGuiVisibility(self,drawGui:bool):
        self.GuiVisible=drawGui
    @typechecked
    def setRenderingVisibility(self,renderVisible:bool):
        self.renderVisible=renderVisible
        
    @typechecked
    def appendGuiCustomization(self, customization: ValueGuiCustomization):
        if customization.valid():
            self.customizations.append (customization) 
    @typechecked
    def getGuiCustomization(self, name:str, type_name:str):
        # Iterate over all customizations and find the one with matching name and type_name
        for customization in self.customizations:
            if customization.name == name and customization.value_type == type_name:
                return customization
        # If no matching customization is found, return None
        return None
    
 
    
    def draw(self):
        if self.renderVisible:
            self.render()        
    @typechecked
    def addAction(self, action_name:str, function: callable):
        # Add a new action to the actions dictionary
        self.actions[action_name] = function

    @typechecked
    def addCallback(self,valueName:str, callback:callable):
        assert(valueName in self.persistentProperties or valueName in self.nonPersistentProperties )
        if valueName in self.callbacks.keys():
            self.callbacks[valueName] = self.callbacks[valueName]+callback
        else:
            self.callbacks[valueName] = [callback] 



    def runAction(self, action_name:str, *args, **kwargs):
        # Execute an action by its name
        if action_name in self.actions:
            return self.actions[action_name](*args, **kwargs)
        else:
            print(f"Action '{action_name}' not found in object '{self.name}'.")
            
    #load_state()  # Load persistent properties from file,called by object manager
    def load_state(self):
        # Load persistent properties from a file
        if  os.path.exists(f'{self.autoSaveFolderPath}/{self.name}.json'):
            with open(f'{self.autoSaveFolderPath}/{self.name}.json', 'r') as file:
                cachedPerperties= jsonpickle.decode(file.read())
                for propName in self.persistentPropertyDefaultValues.keys():
                    if propName in cachedPerperties:
                        self.persistentProperties[propName] = cachedPerperties[propName]                
                    elif  self.persistentPropertyDefaultValues[propName]   is not None:
                        self.persistentProperties[propName] =  self.persistentPropertyDefaultValues[propName]                    
        else:
            print(f"No saved state file \'{self.autoSaveFolderPath}/{self.name}.json\', for {self.name}, skip restoring state.")
            
    def save_state(self):
        """save object state to file
        """
        if self.persistentProperties.keys().__len__()>0:
            # Define the full path for the save file
            file_path = f'{self.autoSaveFolderPath}/{self.name}.json'
            
            # Check if the directory exists, if not, create it
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save persistent properties to a file
            with open(file_path, 'w') as file:
                file.write(jsonpickle.encode(self.persistentProperties))
                
   
    def create_variable_gui(self, name:str, value:any, persistent:bool=False, customizationsParamter:Dict[str,Any]=None,default_value=None) -> None:
        self.create_variable(name, value, persistent, default_value)
        if customizationsParamter is not None:
            cust=ValueGuiCustomization(name,getTypeName(value),customizationsParamter)
            self.appendGuiCustomization(cust)
            
    @typechecked
    def create_variable(self, name:str, value:any, persistent:bool=False, default_value=None) -> None:
         #refactor new_value to same type as value
        if isinstance(value, tuple):
            value = list(value)
            default_value = list(default_value) if default_value is not None else None
        elif getTypeName(value)=="options":
            self.optionValues[name]=value[0]
        # Create a new variable, persistent or non-persistent        
        if persistent:            
            if name not in self.persistentPropertyDefaultValues:  # Set default if not exist                
                self.persistentPropertyDefaultValues[name] = default_value                
            self.persistentProperties[name] = value            
        else:            
            self.nonPersistentProperties[name] = value

    def create_variable_callback(self, name:str, value:any, callback:callable,persistent:bool=False, default_value=None) -> None:
        self.create_variable(name, value, persistent, default_value)
        self.addCallback(name,callback)

    def setValue(self, name:str, value):
        self.updateValue(name, value)

    @typechecked        
    def getOptionValue(self, name:str)->str|None:
            return self.optionValues[name]  if name in self.optionValues else None

    @typechecked
    def updateOptionValue(self, name:str, value:str) -> None:
        self.optionValues[name]=value
        #if has update callback
        callback=self.callbacks.get(name)
        if isinstance(callback, list):
            for cb in callback:
                cb(self)
            


    def updateValue(self, name:str, value:Any)->None:
        # Override to catch properties being set directly
        if name in self.persistentProperties:
            assert(type(value)==type(self.persistentProperties[name]))
            self.persistentProperties[name] = value            
        elif name in self.nonPersistentProperties:
            assert(type(value)==type(self.nonPersistentProperties[name]))
            self.nonPersistentProperties[name] = value 
        else:
            # If the attribute is not found, raise AttributeError
            raise AttributeError(f"'{self.name}' object has no attribute '{name}'")
        #if has update callback
        callback=self.callbacks.get(name)
        if isinstance(callback, list):
            for cb in callback:
                cb(self)

    @typechecked
    def getValue(self, name:str):
        # Override to handle custom property retrieval
        if name in self.persistentProperties:
            return self.persistentProperties[name]
        elif name in self.nonPersistentProperties:
            return self.nonPersistentProperties[name]
        else:
            # If the attribute is not found, raise AttributeError
            raise AttributeError(f"'{self.name}' object has no attribute '{name}'")
    @typechecked
    def hasValue(self, name:str)->bool:
        return name in self.persistentProperties or name in self.nonPersistentProperties



    def DrawPropertiesInGui(self,propertyMap:dict[str, Any],parentNamelist:list=None) -> None:        
        for key, value in propertyMap.items():
            typeName=getTypeName(value)#MAP PYTHON TYPE TO IMGUI TYPE NAME(KEY to get imgui function)
            if isinstance(value, dict):
                # flag= imgui.TREE_NODE_DEFAULT_OPEN|imgui.TREE_NODE_LEAF if noSonDictionary(value) else imgui.TREE_NODE_DEFAULT_OPEN
                expanded = imgui.tree_node(key,imgui.TREE_NODE_DEFAULT_OPEN)   
                if expanded:
                    parentNamelist=[key] if parentNamelist==None  else parentNamelist+[key]
                    self.DrawPropertiesInGui(value,parentNamelist=parentNamelist)
                    imgui.tree_pop() 
            elif  typeName=="options": 
                    callableF= get_imgui_widget_for_type(typeName)
                    changed, new_value = callableF(key,value,self.getOptionValue(key))
                    if changed:
                        self.updateOptionValue(key, new_value)
            else:
                cust=self.getGuiCustomization(key,typeName)
                callableF= get_imgui_widget_for_type(typeName,cust)
                changed, new_value = callableF(key,value)
                if changed:
                    if parentNamelist==None:
                        self.updateValue(key, new_value)
                    else:
                        #parentName is a list of keys to reach the parent dictionary                           
                        valueDictToOperate=self.getValue(parentNamelist[0]) 
                        operate_on_dict(valueDictToOperate,parentNamelist[1:]+[key],new_value,0)
         
    def DrawActionButtons(self) -> None:
        """        
        Draw buttons for all actions defined in the provided object.
        When a button is clicked, the corresponding action is executed.

        :param obj: The object containing actions.
        """
        # Iterate over all actions and create a button for each
        for action_name in self.actions.keys():
            # Draw the button and check if it's clicked
            if imgui.button(f"{action_name}"):
                # Execute the associated action
                self.runAction(action_name,self)
     
    def drawGui(self):
        """Overwrite this function to change the content get rendered  in the ImGui window.
        """
        if self.GuiVisible:
            _,self.GuiVisible=imgui.begin(self.name,self.GuiVisible)
            # Draw persistent properties with appropriate ImGui controls
            self.DrawPropertiesInGui(self.persistentProperties)
            self.DrawPropertiesInGui(self.nonPersistentProperties)
            self.DrawActionButtons()
            imgui.end()

   
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
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d- %(message)s')      

  
   
        # Configure logging to display all messages

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

         
def getLoggingWidget() -> LoggingWidget:
    return LoggingWidget("Logger")

@singleton
class Scene(Object):
    def __init__(self, name,autoSaveFolderPath="autosave"):
        super().__init__(name,autoSaveFolderPath)
        self.objects = {}


    def add_object(self, obj):
        """connect an object with current scene
        """
        assert isinstance(obj,Object)
        if self.hasObject(obj.name):
            logging.getLogger().warning(f" Object '{obj.name}' already exists in scene '{self.name}'.")
            return  
        self.objects[obj.name]= obj
        obj.setUpScene(self)

    def setUpCamera(self, camera):
        for obj in self.objects.values():
                obj.setCamera(camera)
    
    def hasObject(self, name):
        return name in self.objects.keys()
    
    def getObject(self, name): 
        return self.objects.get(name, None)    
    
  
    def draw_all(self):
        for obj in self.objects.values():
            obj.draw()
            
    def save_state_all(self):
        self.save_state()
        for obj in self.objects.values():
            obj.save_state()
            
    def restore_state_all(self):
        self.load_state()
        for obj in self.objects.values():
            obj.load_state()
    
    def toggle_object_visibility(self,ObjectName:str):
        """toggle an onbject's visibility in gui 
        Returns:
            None: 
        """        
        obj=self.getObject(ObjectName)  
        obj.setGuiVisibility(not obj.drawGui) if obj else None     

    def drawGui(self):
        #draw the scene as an object(draw its properties, if any) 
        if imgui.begin(self.name):
            self.DrawPropertiesInGui(self.persistentProperties)
            self.DrawPropertiesInGui(self.nonPersistentProperties)
            self.DrawActionButtons()
            expanded, visible =imgui.collapsing_header("Objects List", flags=imgui.TREE_NODE_DEFAULT_OPEN)
            if expanded:
                for obj_name in self.objects:
                    # This variable tracks the visibility; you might need to store visibility state elsewhere
                    # For demonstration, using a local variable. Consider adapting it to your object's properties
                    obj=self.getObject(obj_name)
                    if isinstance(obj,Object) is False:
                        continue
                    visible = obj.GuiVisible
                    imgui.text(obj_name)
                    imgui.same_line()
                    changed, new_visible = imgui.checkbox(f"##{obj_name}", visible)
                    if changed:
                        obj.setGuiVisibility(new_visible) 
            imgui.end()
        #draw all the objects in the scene
        for obj in self.objects.values():
            obj.drawGui()
            
        

    
    
class TestObject(unittest.TestCase):
    def setUp(self):
        #minic  create object in the first run and restore in the second run
        self.obj = Object('TestObject', autoSaveFolderPath = "autosave")
        self.obj.create_variable('test_key', 'test_value', True, 'default_value')
        self.obj.create_variable('persist_key', 'persist_value', True, 'default0')
        self.obj.create_variable('non_persist_key', 'non_persist_value', False, 'non_persist_key_default1')
    
    def test_singleton(self):  
        @singleton
        class Cls(object):
            def __init__(self):
                pass    
        cls1 = Cls()
        cls2 = Cls()
        self.assertEqual(id(cls1), id(cls2), " Singleton pattern is wrong")

    def test_load_save_state(self):        
        self.obj.save_state()
        #second run
        self.obj = Object('TestObject', autoSaveFolderPath = "autosave")
        self.obj.create_variable('test_key', 'test_value', True, 'default_value')
        self.obj.create_variable('test_key_not_exist_in_cache', 'test_value', True, 80)
        self.obj.create_variable('test_key_noDefaultValue', 'goldValue', True)
        self.obj.load_state()
        self.assertEqual(self.obj.getValue('test_key_noDefaultValue'), 'goldValue', "Should keep persistent value correctly as no defaultValue defined.")
        self.assertEqual(self.obj.getValue('test_key'), 'test_value', "Should restore persistent value correctly.")
        self.assertEqual(self.obj.getValue('test_key_not_exist_in_cache'), 80, "Should not restore persistent value that don't exist in cache file .")
        
    def test_name_attribute(self):
        self.assertEqual(self.obj.name, 'TestObject', "Object name should be correctly set and retrievable.")

    def test_get_value_persistent(self):
        # Check persistent value
        self.assertEqual(self.obj.getValue('persist_key'), 'persist_value', "Should retrieve persistent value correctly.")

    def test_get_value_non_persistent(self):
        # Check non-persistent value
        self.assertEqual(self.obj.getValue('non_persist_key'), 'non_persist_value', "Should retrieve non-persistent value correctly.")

    def test_update_and_get_value(self):
        self.obj.updateValue('persist_key', 'new_value')
        self.assertEqual(self.obj.getValue('persist_key'), 'new_value', "Should update and return new persistent value.")

    def test_has_value(self):
        self.assertTrue(self.obj.hasValue('persist_key'), "Object should report it has the persistent key.")
        self.assertFalse(self.obj.hasValue('some_random_key'), "Object should report it does not have a non-existent key.")


class TestScene(unittest.TestCase):
    def setUp(self):
        self.scene = Scene('TestScene')

    def test_singleton_pattern(self):
        another_scene = Scene('AnotherScene')
        self.assertEqual(id(self.scene), id(another_scene), "Scene should implement the Singleton pattern.")

    def test_add_object(self):
        test_object = Object('TestObject')
        self.scene.add_object(test_object)
        self.assertTrue(self.scene.hasObject(test_object.name), "Scene should contain the added object.")
    def test_scene_name(self):
        self.assertEqual(self.scene.name, 'TestScene', "Scene should have the correct name.")

    def test_scene_restore(self):
        # Assuming a scene can have persistent properties as well
        test_scene = Scene('TestScene', autoSaveFolderPath="autosave")
        test_scene.create_variable('scene_key', 'scene_value', True, 'default')
        self.assertEqual(test_scene.getValue('scene_key'), 'scene_value', "scene_key's value should be scene_value.")
        test_scene.load_state()  # Assuming scene can load its state similar to objects
        self.assertEqual(test_scene.getValue('scene_key'), 'scene_value', "no cache file,value should stay unchanged.")






if __name__ == '__main__':
    unittest.main()


