import jsonpickle
import OpenGL.GL as gl
import numpy as np
import unittest
import os
import imgui
from typing import Dict, Any
from typeguard import typechecked

def singleton(cls):
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return inner

def input_vecn_int_float(label, vecn):
    # Simulate a vec3 input using three separate float input fields.
    # label: The label to display for the vec3 input
    # vec3: A list or tuple containing three float numbers representing the x, y, z components of the vector
    changed = False  # Flag to track if there was a change
    n=len(vecn)
    imgui.text(label)
    if all(isinstance(x, float) for x in vecn):
        for i in range(n):
            imgui.same_line()  # Keep the next input on the same line
            changed_iterm, vecn[i] = imgui.input_float(f"{label}[{i}]", vecn[i])
            changed = changed_iterm or changed   
    else:
        for i in range(n):
            imgui.same_line()  # Keep the next input on the same line
            changed_iterm, vecn[i] = imgui.input_int(f"{label}[{i}]", vecn[i])
            changed = changed_iterm or changed   
    return changed, vecn
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

valid_customizations = {
            'float': [{'widget': 'slider_float', 'min': 0.0, 'max': 1.0}, {'widget': 'input'}],
            'int': [{'widget': 'input'}, {'widget': 'checkbox'}],
            'vec3': [{'widget': 'color_picker'}, {'widget': 'input'}],
            'vec4': [{'widget': 'color_picker'}, {'widget': 'input'}],
            'ivec3': [{'widget': 'input'},{'widget': 'color_picker'}],
            'ivec4': [{'widget': 'input'},{'widget': 'color_picker'}],
            'bool': [{'widget': 'checkbox'}],
            'str': [{'widget': 'input'}],
            'ivecn': [{'widget': 'input'}],
            'vecn': [{'widget': 'input'},{'widget': 'plot_lines'}],
            'options':[{'widget': 'combo'}]
            # Add more types as needed
        }



def getTypeName(value) -> Any | str:
    if isinstance(value, bool) :
        return "bool"
    elif isinstance(value, (list, tuple)) and all(isinstance(x, bool) for x in value):
        return "bvecn"
    elif isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(x, float) for x in value):
        return "vec3"
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(x, int) for x in value):
        return "ivec3"
    elif isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(x, float) for x in value):
        return "vec4"
    elif isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(x, int) for x in value):
        return "ivec4"
    elif isinstance(value, (list, tuple)) and all(isinstance(x, int) for x in value):
        return "ivecn"
    elif isinstance(value, (list, tuple)) and all(isinstance(x, float) for x in value):
        return "vecn"
    elif isinstance(value, list)  and all(isinstance(x, str) for x in value):
        return "options"
    else:
        return type(value).__name__
        
                    
def draw_combo_options(label:str, x_list:list,current_selection_i:str):
    # Draw a combo box with the provided options and return the selected index
    current_selection = current_selection_i if current_selection_i else x_list[0]
    clicked=False
    if current_selection is not None and imgui.begin_combo(label, current_selection):
        for x in x_list:
            clicked_this, _ = imgui.selectable(x, x == current_selection)
            if clicked_this:
                current_selection = x
                clicked=True
        imgui.end_combo()
    
    return clicked, current_selection

class ValueGuiCustomization:
    '''ValueGuiCustomization class to store the customization parameters for draw the gui for a value;
    The usage is to create a ValueGuiCustomization object with "name" "type" and append it to the an Object .
    The Object's propertie with that name and type will be drawn in the gui with the customization parameters.
    Mismatch in name or type will make the valueguicustomization get ignored.
    '''
    def __init__(self,name,value_type,customizationsParamter:Dict[str,Any]):
        self.name = name
        self.value_type = value_type
        if  self.check_customization_parameters(customizationsParamter):
            self.customizationsParamter = customizationsParamter
        else: 
            self.customizationsParamter = None
        
    def valid(self):
        return self.customizationsParamter is not None
    
    def get_customization(self)->Dict[str,Any]:
        return  self.customizationsParamter 
    def get(self,key:str)->Any:
        return self.customizationsParamter[key]
    
    def check_customization_parameters(self,customDict:Dict[str,Any]):
        valid_params = valid_customizations.get(self.value_type, [])        
        # Check if any of valid customization options match the provided customization parameters
        for option in valid_params:
            keys1 = set(option.keys())
            keys2 = set(customDict.keys())
            if keys1 == keys2:
                return True
        else:
            print(f"Error: Customization parameters for {self.name} of type {self.value_type} are invalid.")
            return False


def get_imgui_widget_for_type(Value_type:str, customization:ValueGuiCustomization=None):
    """
    Selects and returns the appropriate ImGui widget based on customization type and entry.

    :param customization_type: The type of the customization (e.g., 'float', 'int', 'vec3').
    :param customization_entry: A dictionary representing a customization entry for the type.
    :return: A callable representing the corresponding imgui widget for the type.
    """
    widget_map_by_type = {
        'bool': {         
            'checkbox': imgui.checkbox,
        },
        'float': {
             'input': imgui.input_float,
            'slider_float': lambda label, value: imgui.slider_float(label, value, customization.get('min') , customization.get('max')),
        },
        'int': {
            'input': imgui.input_int,
            'checkbox':  imgui.checkbox,
        },
         'ivec3': {
            'input': lambda label, value: imgui.drag_int3(label, *value),
             
        },
        'ivec4': {
            'input': lambda label, value:  imgui.drag_int3(label, *value), 
        },
        'vec3': {  
            'input': lambda label, value: input_vecn_int_float(label, list(value)) , 
            'color_picker': lambda label, value:  imgui.color_edit3(label, *value), 
        },
        'vec4': {
             'input': lambda label, value: imgui.input_vec4(label,*value),
            'color_picker':lambda label, value:  imgui.color_edit4(label, *value), 
        },
         'vecn': {  
            'input': lambda label, value: input_vecn_int_float(label, list(value)) , 
            'plot_lines': lambda label, value: imgui.plot_lines(label, np.array(value,dtype=np.float32)) , 
        },
        'ivecn': {  
            'input': lambda label, value: input_vecn_int_float(label, list(value)) , 
        },
        'str': {
            'input': lambda label,value:imgui.input_text(label, value, 256),
        },
        'options': {
            'combo': lambda label,value_list,current_selection:draw_combo_options(label, value_list, current_selection)
        },
        # add more
    }

    widget_map = widget_map_by_type.get(Value_type)
    if widget_map:
        callable=widget_map.get(customization.get('widget'), None) if customization else None
        if callable is None:
            first_key = next(iter(widget_map))
            callable=widget_map[first_key]
        return callable
    print(f"Error: No ImGui widget found for  '{customization.value_type}' '{customization.name}' .")
    return   imgui.text


class Object:
    def __init__(self, name:str,autoSaveFolderPath = "autosave"):
        self.name = name
        self.persistentPropertyDefaultValues={}
        self.persistentProperties = {}
        self.nonPersistentProperties = {}
        self.autoSaveFolderPath =autoSaveFolderPath
        self.customizations=[]
        self.actions = {}
        self.optionValues = {}
        self.GuiVisible=True
        self.renderVisible=False
        
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
    
    def render(self):
        pass
    
    def draw(self):
        if self.renderVisible:
            self.render()        
    @typechecked
    def addAction(self, action_name:str, function: callable):
        # Add a new action to the actions dictionary
        self.actions[action_name] = function

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
        # Create a new variable, persistent or non-persistent
        if persistent:            
            if name not in self.persistentPropertyDefaultValues:  # Set default if not exist                
                self.persistentPropertyDefaultValues[name] = default_value                
            self.persistentProperties[name] = value            
        else:            
            self.nonPersistentProperties[name] = value
    def setValue(self, name:str, value):
        self.updateValue(name, value)
    def getOptionValue(self, name:str)->str|None:
            return self.optionValues[name]  if name in self.optionValues else None

    def updateOptionValue(self, name:str, value) -> None:
         self.optionValues[name]=value


    def updateValue(self, name:str, value):
  
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
                if  cust is not None and cust.get('widget')=='plot_lines':
                    callableF(key,value)  
                else:
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
            imgui.begin(self.name)
            # Draw persistent properties with appropriate ImGui controls
            self.DrawPropertiesInGui(self.persistentProperties)
            self.DrawPropertiesInGui(self.nonPersistentProperties)
            self.DrawActionButtons()
            imgui.end()
       

@singleton
class Scene(Object):
    def __init__(self, name,autoSaveFolderPath="autosave"):
        super().__init__(name,autoSaveFolderPath)
        self.objects = {}
    def add_object(self, obj):
        self.objects[obj.name]= obj
    def hasObject(self, name):
        return name in self.objects.keys()
    def getObject(self, name): 
        return self.objects.get(name, None)    
    def drawGui(self):
        super().drawGui()
        for obj in self.objects.values():
            obj.drawGui()
            
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
            _type_: _description_
        """        
        obj=self.getObject(ObjectName)  
        obj.setGuiVisibility(not obj.drawGui) if obj else None     
        

    
    
    
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


