import imgui
from typing import Dict, Any
import numpy as np
import ctypes
import logging
###
###all gui widget functions
###
def draw_bool_checkbox( label, value):
    changed, new_value = imgui.checkbox(label, value)
    return changed, new_value

def draw_uint_input(label, value, min_value=0, max_value=4294967295):
    """
    Custom widget to handle uint (unsigned integer) values in ImGui.

    Args:
        label (str): The label for the widget.
        value (ctypes.c_uint): The uint value to be displayed and edited.
        min_value (int, optional): The minimum value for the uint. Defaults to 0.
        max_value (int, optional): The maximum value for the uint. Defaults to 4294967295 (max value for 32-bit uint).

    Returns:
        bool, ctypes.c_uint: A tuple containing a boolean indicating if the value was changed, and the updated uint value.
    """
    changed, value_int = imgui.input_int(label, int(value))
    value_int = max(min_value, min(max_value, value_int))
    value = ctypes.c_uint(value_int)
    return changed, value

def draw_float_input(label, value):
    changed, new_value = imgui.input_float(label, value)
    return changed, new_value


def draw_float_slider( label, value,customization):
    changed, new_value = imgui.slider_float(label, value, customization.get('min'), customization.get('max'))
    return changed, new_value


def draw_int_input( label, value):
    changed, new_value = imgui.input_int(label, value)
    return changed, new_value

def draw_int_checkbox( label, value):
    changed, new_value = imgui.checkbox(label, value)
    return changed, int(new_value)

def draw_ivec3_drag( label, value):
    changed, new_value = imgui.drag_int3(label, *value)
    return changed, list(new_value) 


def draw_ivec4_drag( label, value):
    changed, new_value = imgui.drag_int4(label, *value)
    return changed, list(new_value) 

def draw_color_edit3( label, value):
    changed, new_value = imgui.color_edit3(label, *value)
    return changed, list(new_value) 


def draw_vec4_input( label, value):
    changed, new_value = imgui.input_vec4(label, *value)
    return changed, list(new_value) 


def draw_color_edit4( label, value):
    changed, new_value = imgui.color_edit4(label, *value)
    return changed, list(new_value) 

def draw_numpy_array3_input( label, arr):
    """
    Draw a NumPy array of shape (3,) into an ImGui window.
    
    Args:
        Obj (Object): The object to update the value for.
        label (str): The label to display for the array.
        arr (np.ndarray): The NumPy array of shape (43,) to be displayed.
    """
    if len(arr) != 3:
        raise ValueError("The input array must have a shape of (3,)")

    value = arr.copy()
    changed, new_value = imgui.drag_float3(label, *value)
    return changed, np.array(new_value,dtype=arr.dtype)

def draw_numpy_array4_input( label, arr):
    """
    Draw a NumPy array of shape (4,) into an ImGui window.
    
    Args:
        Obj (Object): The object to update the value for.
        label (str): The label to display for the array.
        arr (np.ndarray): The NumPy array of shape (4,) to be displayed.
    """
    if len(arr) != 4:
        raise ValueError("The input array must have a shape of (4,)")

    value = arr.copy()
    changed, new_value = imgui.drag_float4(label, *value)
    return changed, np.array(new_value,dtype=arr.dtype)


def draw_vecn_input( label, value):
    changed, new_value = input_vecn_int_float(label, value)
    return changed, list(new_value) 



def draw_vecn_plot_lines( label, value):
    imgui.plot_lines(label, np.array(value,dtype=np.float32))
    return False, None


def draw_ivecn_input( label, value):
    changed, new_value =input_vecn_int_float(label, value)
    return changed,  list(new_value) 

def draw_str_input( label, value):
    changed, new_value = imgui.input_text(label, value, 256)
    return changed, new_value


def draw_options_combo( label, value_list, current_selection):
    def draw_combo_options(label:str, x_list:list,current_selection_i:str):
        # Draw a combo box with the provided options and return the selected index
        current_selection = current_selection_i
        clicked=False
        if current_selection is not None and imgui.begin_combo(label, current_selection):
            for x in x_list:
                clicked_this, _ = imgui.selectable(x, x == current_selection)
                if clicked_this:
                    current_selection = x
                    clicked=True
            imgui.end_combo()
        
        return clicked, current_selection

    changed, newSlection = draw_combo_options(label, value_list, current_selection)
    return changed, newSlection

def draw_numpy_array(key, array):
    if not isinstance(array, np.ndarray):
        print("Error: Input is not a numpy array")
        return False, None
    imgui.text(key+":")  
    for row_idx, row in enumerate(array):
        row_str = " ".join(str(x) for x in row) 
        imgui.text(f"Row {row_idx}: {row_str}")  
    return False, None
                    
def draw_editable_mat4(label, mat):
 
    changed = False
    imgui.text(label+":")  
    imgui.push_item_width(50)
    for row in range(4):
        if row > 0:
            imgui.spacing()
        for column in range(4):
            elem_label = f"##{label}_{row}_{column}"
            _, mat[row, column] = imgui.input_float(elem_label, mat[row, column])
            if _:
                changed = True
     
            if column < 3:
                imgui.same_line()
    imgui.pop_item_width()
    return changed, mat

def input_vecn_int_float(label, vecn_input):
    # Simulate a vec3 input using three separate float input fields.
    # label: The label to display for the vec3 input
    # vec3: A list or tuple containing three float numbers representing the x, y, z components of the vector
    changed = False  # Flag to track if there was a change
    vecn=list(vecn_input)
    n=len(vecn)
    imgui.push_item_width(80)
    imgui.text(label+":")
    # imgui.same_line()
    if all(isinstance(x, float) for x in vecn):
        for i in range(n):
            changed_iterm, vecn[i] = imgui.drag_float(f"{label}[{i}]", vecn[i],)
            changed = changed_iterm or changed   
            if i < n - 1:
                imgui.same_line()
    else:
        for i in range(n):
            changed_iterm, vecn[i] = imgui.drag_float(f"{label}[{i}]", vecn[i])
            changed = changed_iterm or changed   
            if i < n - 1:
                imgui.same_line()
    imgui.pop_item_width()
    return changed, vecn




valid_customizations = {
            'float': [{'widget': 'slider_float', 'min': 0.0, 'max': 1.0}, {'widget': 'input'}],
            'uintc': [{'widget': 'input'}],
            'int': [{'widget': 'input'}, {'widget': 'checkbox'}],
            'vec3': [{'widget': 'color_picker'}, {'widget': 'input'}],
            'vec4': [{'widget': 'color_picker'}, {'widget': 'input'}],
            'np.vec3': [ {'widget': 'input'}],
            'np.vec4': [ {'widget': 'input'}],
            'np.mat4': [ {'widget': 'input'}],
            'ndarray': [ {'widget': 'input'}],
            'ivec3': [{'widget': 'input'},{'widget': 'drag'}],
            'ivec4': [{'widget': 'input'},{'widget': 'drag'}],
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
    elif isinstance(value, np.ndarray) and value.shape == (3,):
        return "np.vec3"
    elif isinstance(value, np.ndarray) and value.shape == (4,):
        return "np.vec4"
    elif isinstance(value, np.ndarray) and value.shape == (4,4):
        return "np.mat4"
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
            'checkbox':lambda  label,value: draw_bool_checkbox(  label, value),
         },
        'float': {
            'input':lambda   label, value: draw_float_input( label, value),
            'slider_float': lambda   label, value: draw_float_slider( label, value,customization),
        },
        'uintc': {
            'input': lambda  label, value: draw_uint_input( label, value),
        },
        'int': {
            'input': lambda  label, value: draw_int_input( label, value),
            'checkbox': lambda  label, value: draw_int_checkbox( label, value),
        },
         'ivec3': {
            'drag': lambda  label, value: draw_ivec3_drag( label, value),
            'input':lambda  label, value: draw_vecn_input( label, value) , 
        },
        'ivec4': {
            'drag': lambda  label, value: draw_ivec4_drag( label, value),
            'input': lambda  label, value: draw_vecn_input( label, value) , 
        },
        'vec3': {  
            'input': lambda  label, value: draw_vecn_input( label, value) , 
            'color_picker': lambda label, value: draw_color_edit3( label, value),
        },
        'vec4': {
            'input': lambda  label, value: draw_vec4_input( label, value),
            'color_picker':lambda  label, value: draw_color_edit4( label, value),
        },
        'np.vec4': {
             'input': lambda  label, value: draw_numpy_array4_input( label, value),
        },
        'np.vec3': {
             'input': lambda  label, value: draw_numpy_array3_input( label, value),
        },
         'ndarray': {
             'input': lambda  label, value: draw_numpy_array(label, value),
        },
         'np.mat4': {
             'input': lambda  label, value: draw_editable_mat4( label, value),
        },
         'vecn': {  
            'input': lambda  label, value: draw_vecn_input( label, value) , 
            'plot_lines': lambda label, value: draw_vecn_plot_lines(label,value) , 
        },
        'ivecn': {  
            'input': lambda  label, value: draw_vecn_input( label, value) , 
        },
        'str': {
            'input': lambda  label, value: draw_str_input( label, value),
        },
        'options': {
            'combo': lambda  label,value_list,current_selection:draw_options_combo(label, value_list, current_selection)
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
    logging.error(f"Error: No ImGui widget found for  '{Value_type}' .")
    return None
      
 
