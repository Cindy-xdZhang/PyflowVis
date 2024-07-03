import numpy as np
from PIL import Image, ImageDraw,ImageFont
import logging
import math
from .VectorField2d import SteadyVectorField2D,UnsteadyVectorField2D
import os
from typeguard import typechecked
def draw_arrow(draw, start, end, color, arrowhead_length=5, arrowhead_angle=10):
    """
    Draw an arrow from start to end with arrowhead.
    
    Parameters:
    - draw: ImageDraw object to draw on.
    - start: tuple, the starting point of the arrow.
    - end: tuple, the ending point of the arrow.
    - color: tuple, the color of the arrow.
    - arrowhead_length: int, the length of the arrowhead.
    - arrowhead_angle: float, the angle of the arrowhead in degrees.
    """
    draw.line([start, end], fill=color, width=1)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    
    # Create the arrowhead
    arrow_angle1 = angle + math.radians(arrowhead_angle)
    arrow_angle2 = angle - math.radians(arrowhead_angle)
    
    arrowhead1 = (end[0] - arrowhead_length * math.cos(arrow_angle1),
                  end[1] - arrowhead_length * math.sin(arrow_angle1))
    arrowhead2 = (end[0] - arrowhead_length * math.cos(arrow_angle2),
                  end[1] - arrowhead_length * math.sin(arrow_angle2))
    
    draw.line([end, arrowhead1], fill=color, width=1)
    draw.line([end, arrowhead2], fill=color, width=1)

def scalar_to_color(scalar, min_scalar, max_scalar):
    """
    Map a scalar value to an RGB color using a simple linear color mapping.
    
    Parameters:
    - scalar: float, the scalar value to map.
    - min_scalar: float, the minimum scalar value in the data.
    - max_scalar: float, the maximum scalar value in the data.
    
    Returns:
    - color: tuple, the RGB color corresponding to the scalar value.
    """
    normalized_scalar = (scalar - min_scalar) / (max_scalar - min_scalar)  # Normalize the scalar to [0, 1]
    # Define a simple linear colormap from blue to red
    blue = (51, 255, 255)
    red = (255, 0, 0)
    r = int(blue[0] * (1 - normalized_scalar) + red[0] * normalized_scalar)
    g = int(blue[1] * (1 - normalized_scalar) + red[1] * normalized_scalar)
    b = int(blue[2] * (1 - normalized_scalar) + red[2] * normalized_scalar)
    return (r, g, b)

@typechecked
def glyphsRenderSteadyFieldAlgorthim(vecfield: SteadyVectorField2D, image_size=(800, 800), ColorCodingFn=lambda u, v: math.sqrt(u*u + v*v),gridSkip=2) -> Image.Image:
    pushLogLevel = logging.getLogger().getEffectiveLevel()
    logger = logging.getLogger()
    logger .setLevel(logging.WARNING)
    Ydim, Xdim, _ = vecfield.field.shape
    img = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(img)
    
    # Define the grid in image coordinates
    img_width, img_height = image_size


    # Compute scalar values for all vectors for color mapping
    scalars = np.zeros((Ydim, Xdim))
    fieldMaginitude= np.zeros((Ydim, Xdim))
    for y in range(Ydim):
        for x in range(Xdim):
            u, v = vecfield.field[y, x]
            scalars[y, x] = ColorCodingFn(u, v)
            fieldMaginitude[y, x]=math.sqrt(u*u+v*v)
    min_scalar = np.min(scalars)
    max_scalar = np.max(scalars)

    min_mag = np.min(fieldMaginitude)
    max_mag = np.max(fieldMaginitude)
    vector_feild_scale=1.0
    if max_mag<=0.1:
        vector_feild_scale=0.1/(max_mag+0.0000001)

    arrayHeadLength=(min(img_width, img_height)/800 )*3
    arrayStemLength=(min(img_width, img_height)/800 )*5
    for y in range(0,Ydim,gridSkip):
        for x in range(0,Xdim,gridSkip):
            # Map grid coordinates to image coordinates
            img_x = int(x * img_width / Xdim)
            img_y = int(y * img_height / Ydim)
            
            # Get the vector at this grid point
            u, v = vecfield.field[y, x]*vector_feild_scale
            
    
            end_x = img_x + int(u * arrayStemLength)
            end_y = img_y + int(v * arrayStemLength)
            
            # Get color from scalar value
            scalar = scalars[y, x]
            color = scalar_to_color(scalar, min_scalar, max_scalar)
            
            draw_arrow(draw, (img_x, img_y), (end_x, end_y), color=color,arrowhead_length=arrayHeadLength)

    minv_text = f"minv={min_mag:.4f}"
    maxval_text = f"maxval={max_mag:.4f}"
    text_color = (255, 0, 0)  # Red color
    font_size=12
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if the specified font is not found

    draw.text((710, 750), minv_text, fill=text_color, font=font)
    draw.text((710, 770), maxval_text, fill=text_color, font=font)
    logger.setLevel(pushLogLevel)
    return img

@typechecked
def glyphsRenderSteadyField(vecfield: SteadyVectorField2D, output_file: str, image_size=(800, 800), ColorCodingFn=lambda u, v: math.sqrt(u*u + v*v),gridSkip=2):
    """
    Render a 2D vector field as vector glyphs using Pillow and save the output as a PNG file.
    
    Parameters:
    - vecfield: SteadyVectorField2D, the 2D vector field to render.
    - output_file: str, the path to the output PNG file.
    - image_size: tuple, the size of the output image.
    - ColorCodingFn: function, a function that maps vector components to a scalar.
    """
    img = glyphsRenderSteadyFieldAlgorthim(vecfield, image_size, ColorCodingFn,gridSkip)
    save_name=output_file if output_file.endswith("png") else f"{output_file}.png"
    img.save(save_name)
   


@typechecked
def glyphsRenderUnsteadyField(field:UnsteadyVectorField2D,ImageSize:int,timeStepSKip:int=2,saveFolder:str="./",saveName:str="vectorGlyph",ColorCodingFn=lambda u, v: math.sqrt(u*u + v*v)):
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    #typecheck field type and field is not None    
    Xdim,Ydim,time_steps=field.Xdim,field.Ydim,field.time_steps
    for i in range(0, time_steps, timeStepSKip):
        steadyVectorField2D = field.getSlice(i)
        output_file = os.path.join(saveFolder, f"{saveName}_{i}.png")
        glyphsRenderSteadyField(steadyVectorField2D, output_file, image_size=(ImageSize, ImageSize), ColorCodingFn=ColorCodingFn)