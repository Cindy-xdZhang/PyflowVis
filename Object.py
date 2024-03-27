import jsonpickle
import OpenGL.GL as gl
import numpy as np
import unittest
import os
import imgui
from typing import Dict, Any

valid_customizations = {
            'float': [{'widget': 'slider_float', 'min': 0.0, 'max': 1.0}, {'widget': 'input_float'}],
            'int': [{'widget': 'input_int'}, {'widget': 'checkbox'}], # add radio_button in future
            'vec4': [{'widget': 'color_picker'}, {'widget': 'drag_float4'}],
            'float_array': [{'widget': 'plot_lines'}],
            'string': [{'widget': 'input_text'}],
            # Add more types as needed
        }

class ValueGuiCustomization:
    '''ValueGuiCustomization class to store the customization parameters for draw the gui for a value;
    The usage is to create a ValueGuiCustomization object with "name" "type" and append it to the an Object .
    The Object's propertie with that name and type will be drawn in the gui with the customization parameters.
    Mismatch in name or type will make the valueguicustomization get ignored.
    '''
    def __init__(self,name,value_type,customizationsParamter:Dict[str,Any]):
        self.name = name
        self.value_type = value_type
        if self.check_customization_parameters(customizationsParamter):
            self.customizationsParamter = customizationsParamter
        else: 
            self.customizationsParamter = None
        
    def valid(self):
        return self.customizationsParamter is not None
    
    def get_customization(self, value_type)->Dict[str,Any]:
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
def input_vec3(label, vec3):
    # Simulate a vec3 input using three separate float input fields.
    # label: The label to display for the vec3 input
    # vec3: A list or tuple containing three float numbers representing the x, y, z components of the vector
    changed = False  # Flag to track if there was a change

    # Begin on the same line
    imgui.text(label)
    imgui.same_line()

    # X component
    changed_x, vec3[0] = imgui.input_float(f"##{label}_x", vec3[0])
    imgui.same_line()  # Keep the next input on the same line

    # Y component
    changed_y, vec3[1] = imgui.input_float(f"##{label}_y", vec3[1])
    imgui.same_line()  # Keep the next input on the same line

    # Z component
    changed_z, vec3[2] = imgui.input_float(f"##{label}_z", vec3[2])

    # If any component was changed, set the changed flag to True
    if changed_x or changed_y or changed_z:
        changed = True

    # Return whether there was a change and the updated vector
    return changed, vec3

def getTypeName(value):
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return "vec3"
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        return "vec4"
    else:
        return type(value).__name__

class Object:
    def __init__(self, name,autoSaveFolderPath = "autosave"):
        self.name = name
        self.persistentPropertyDefaultValues={}
        self.persistentProperties = {}
        self.nonPersistentProperties = {}
        self.autoSaveFolderPath =autoSaveFolderPath
        self.customizations=[]
        self.actions = {}
        self.drawGui=True
    def stDrawGui(self,drawGui:bool):
        self.drawGui=drawGui
        
    def appendGuiCustomization(self, customization: ValueGuiCustomization):
        if customization.valid():
            self.customizations.append (customization) 
    def getGuiCustomization(self, name, type_name):
        # Iterate over all customizations and find the one with matching name and type_name
        for customization in self.customizations:
            if customization.name == name and customization.value_type == type_name:
                return customization
        # If no matching customization is found, return None
        return None
    def draw(self):
        pass  
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


    def create_variable(self, name:str, value:any, persistent:bool, default_value=None):
        # Create a new variable, persistent or non-persistent
        if persistent:            
            if name not in self.persistentPropertyDefaultValues:  # Set default if not exist                
                self.persistentPropertyDefaultValues[name] = default_value                
            self.persistentProperties[name] = value            
        else:            
            self.nonPersistentProperties[name] = value
    def setValue(self, name:str, value):
        self.updateValue(name, value)
    def updateValue(self, name:str, value):
        # Override to catch properties being set directly
        if name in self.persistentProperties:
            self.persistentProperties[name] = value            
        elif name in self.nonPersistentProperties:
            self.nonPersistentProperties[name] = value 
        else:
            # If the attribute is not found, raise AttributeError
            raise AttributeError(f"'{self.name}' object has no attribute '{name}'")

    def getValue(self, name:str):
        # Override to handle custom property retrieval
        if name in self.persistentProperties:
            return self.persistentProperties[name]
        elif name in self.nonPersistentProperties:
            return self.nonPersistentProperties[name]
        else:
            # If the attribute is not found, raise AttributeError
            raise AttributeError(f"'{self.name}' object has no attribute '{name}'")
    def hasValue(self, name:str):
        return name in self.persistentProperties or name in self.nonPersistentProperties
    
    def DrawPropertiesInGui(self,propertyMap:dict[str, Any],level=0):        
            for key, value in propertyMap.items():
                typeName=getTypeName(value)
                cust=self.getGuiCustomization(key,typeName)
                if isinstance(value, dict):
                    expanded = imgui.tree_node(key, imgui.TREE_NODE_DEFAULT_OPEN)
                    if expanded:
                        self.DrawPropertiesInGui(value,level=level+1)
                        imgui.tree_pop() if level==0 else None
                # Check type of value and render the appropriate ImGui widget
                elif isinstance(value, float):
                    if  cust is not None and cust.get('widget') == 'slider_float':
                        changed, new_value = imgui.slider_float(key, value, cust['min'], cust['max'])
                        if changed:
                            self.updateValue(key, new_value)                    
                    elif cust is None or cust.get('widget') == 'input_float' :
                        changed, new_value = imgui.input_float(key, value)
                        if changed:
                            self.updateValue(key, new_value)
                    else:
                        print(f"Error: Invalid customization for {key} of type {type(value).__name__}.")      
                elif isinstance(value, int):
                    # Integer value, use input int
                    if  cust is not None and cust.get('widget') == 'checkbox' :
                        changed, new_value = imgui.checkbox(key, bool(value))
                        if changed:
                            self.updateValue(key, new_value)
                    elif cust is  None or cust.get('widget') == 'input_int':
                        changed, new_value = imgui.input_int(key, value)
                        if changed:
                            self.updateValue(key, new_value)
                    else:
                        print(f"Error: Invalid customization for {key} of type {type(value).__name__}.")
                elif isinstance(value, (list, tuple)) and len(value) == 3:
                    changed, new_value = input_vec3(key, list(value))
                    if changed:
                            self.updateValue(key, new_value)
                elif isinstance(value, (list, tuple)) and len(value) == 4:
                    if  cust is not None and cust.get('widget') == 'color_picker': 
                        changed, new_value = imgui.color_edit4(key, *value)
                        if changed:
                            self.updateValue(key, new_value)
                    elif cust is None or cust.get('widget') == 'drag_float4':
                        changed, new_value = imgui.drag_float4(f"##{key}", *value)
                        if changed:
                            self.updateValue(key, new_value)
                    else:
                         print(f"Error: Invalid customization for {key} of type {type(value).__name__}.")
            
                elif isinstance(value, str):
                    buffer_size = 256  # maximum number of characters
                    input_changed, input_value = imgui.input_text(key, value, buffer_size)
                    if input_changed:
                        self.updateValue(key, input_value)
                else:
                    # For other types, use text for now
                    imgui.text(f"{key}: {value}")
        
    def DrawActionButtons(self):
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

            
            
        
    def DrawGui(self):
        if imgui.begin(self.name):
            # Draw persistent properties with appropriate ImGui controls
            self.DrawPropertiesInGui(self.persistentProperties)
            self.DrawPropertiesInGui(self.nonPersistentProperties)
            self.DrawActionButtons()
            imgui.end()
       


class Scene(Object):
    _instance = None
    def __init__(self, name,autoSaveFolderPath="autosave"):
        self.objects = {}
    def add_object(self, obj):
        self.objects[obj.name]= obj
    def hasObject(self, name):
        return name in self.objects.keys()
    def DrawGui(self):
        super().DrawGui()
        for obj in self.objects.values():
            obj.DrawGui()
            
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
            
    def __new__(cls, name='Scene', autoSaveFolderPath="autosave"):
        if cls._instance is None:
            cls._instance = super(Scene, cls).__new__(cls)
            # Initialize the instance once:
            Object.__init__(cls._instance, name,autoSaveFolderPath=autoSaveFolderPath)            
        return cls._instance
    
    
    
class TestObject(unittest.TestCase):
    def setUp(self):
        #minic  create object in the first run and restore in the second run
        self.obj = Object('TestObject', autoSaveFolderPath = "autosave")
        self.obj.create_variable('test_key', 'test_value', True, 'default_value')
        self.obj.create_variable('persist_key', 'persist_value', True, 'default0')
        self.obj.create_variable('non_persist_key', 'non_persist_value', False, 'non_persist_key_default1')
    
        
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


