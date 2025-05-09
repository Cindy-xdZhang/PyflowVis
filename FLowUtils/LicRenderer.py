import numpy as np
from .VectorField2d import *
from typeguard import typechecked
import os,math
from PIL import Image
from .GlyphRenderer import glyphsRenderSteadyFieldAlgorthim
from .interpolation import bilinear_interpolate
from .Pyds.CppPlugins import cppMoudules      

def LICAlgorithm(texture:np.ndarray, vecfield: SteadyVectorField2D, resultImageSizeX, resultImageSizeY,stepSize:float, MaxIntegrationSteps:int):
    """
    A simplified LIC algorithm to visualize the flow of a 2D vector field slice. 
    texture as same size as the vector field slice, vecfield is the vector field slice,
    resultImageSizeX and resultImageSizeY are the size of the output image, 
    stepSize is the step size for integration, and MaxIntegrationSteps is the maximum number of integration steps to take.
    """
    if texture.ndim == 2:
        Ydim, Xdim = texture.shape
        texture = texture[:, :, np.newaxis]  # Add a dummy channel dimension
    else:
        Ydim, Xdim, _ = texture.shape

    output_texture = np.zeros((resultImageSizeY, resultImageSizeX), dtype=np.float32)
    vecfieldData=vecfield.field
    domainRangeX=vecfield.domainMaxBoundary[0]-vecfield.domainMinBoundary[0]
    domainRangeY=vecfield.domainMaxBoundary[1]-vecfield.domainMinBoundary[1]

    inverse_grid_interval_x=1/float(vecfield.gridInterval[0])
    inverse_grid_interval_y=1/float(vecfield.gridInterval[1])
    for y in range(resultImageSizeY):
        for x in range(resultImageSizeX):

            ratioX=float(x)/float(resultImageSizeX)
            ratioY=float(y)/float(resultImageSizeY)
            accum_value = 0.0
            accum_count = 0
            
            # Trace forward
            #pos (x,y)
            pos = np.array([ratioX* domainRangeX+vecfield.domainMinBoundary[0], ratioY* domainRangeY+vecfield.domainMinBoundary[1]], dtype=np.float32)

            for _ in range(MaxIntegrationSteps):
                floatIndexX=(pos[0]-vecfield.domainMinBoundary[0])*inverse_grid_interval_x
                floatIndexY=(pos[1]-vecfield.domainMinBoundary[1])*inverse_grid_interval_y
                if not (0 <= floatIndexX < Xdim and 0 <= floatIndexY < Ydim):
                    break  # Stop if we move outside the texture bounds

                accum_value += bilinear_interpolate(texture, floatIndexX, floatIndexY)
                accum_count += 1
                vec =bilinear_interpolate(vecfieldData,  floatIndexX, floatIndexY)
                pos += vec * stepSize
                
            # Trace backward
            pos = np.array([ratioX* domainRangeX+vecfield.domainMinBoundary[0], ratioY* domainRangeY+vecfield.domainMinBoundary[1]], dtype=np.float32)
            for _ in range(MaxIntegrationSteps):
                floatIndexX=(pos[0]-vecfield.domainMinBoundary[0])*inverse_grid_interval_x
                floatIndexY=(pos[1]-vecfield.domainMinBoundary[1])*inverse_grid_interval_y
                if not (0 <= floatIndexX < Xdim and 0 <= floatIndexY < Ydim):
                    break  # Stop if we move outside the texture bounds

                accum_value += bilinear_interpolate(texture, floatIndexX, floatIndexY)
                accum_count += 1
                vec =bilinear_interpolate(vecfieldData,  floatIndexX, floatIndexY)

                pos -= vec * stepSize
            
            # Compute the average value along the path
            if accum_count > 0:
                output_texture[y, x] = accum_value / accum_count
    
    return output_texture



@typechecked
def LicRenderingSteady(vecfield: SteadyVectorField2D,licImageSize:int,saveFolder:str="./",saveName:str="vector_field_lic",stepSize=0.01, MaxIntegrationSteps=128):
    """
    Render a steady 2D vector field as an LIC image and save to a PNG file.
    """ 
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    # Step 1: Initialize a texture for the LIC process, often random noise
    texture = np.random.rand(vecfield.Ydim, vecfield.Xdim)

    lic_result = LICAlgorithm(texture, vecfield, licImageSize,licImageSize,stepSize, MaxIntegrationSteps)
    
    # Step 3: Normalize the LIC result for visualization
    lic_normalized = (lic_result - np.min(lic_result)) / (np.max(lic_result) - np.min(lic_result))
    
    # Step 4: Convert to an image and save
    lic_normalized_img = (lic_normalized * 255).astype(np.uint8)  # Convert to 8-bit grayscale
    img = Image.fromarray(lic_normalized_img, mode='L')

    save_name=saveName if saveName.endswith("png") else f"{saveName}.png"
    savePath=os.path.join(saveFolder,save_name)
    img.save(savePath)


@typechecked
def LicRenderingUnsteady(field:UnsteadyVectorField2D,licImageSize:int,timeStepSKip:int=2,saveFolder:str="./",saveName:str="vector_field_lic",stepSize=0.01, MaxIntegrationSteps=128):
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    #typecheck field type and field is not None    
    Xdim,Ydim,time_steps=field.Xdim,field.Ydim,field.time_steps
    texture = np.random.rand(Xdim, Ydim)    
    for i in range(0, time_steps, timeStepSKip):
        print(f"Processing time step {i}")
        steadyVectorField2D = field.getSlice(i)
        lic_result=LICAlgorithm(  texture  ,steadyVectorField2D ,licImageSize,licImageSize,stepSize,MaxIntegrationSteps)
        lic_normalized =255* (lic_result - np.min(lic_result)) / (np.max(lic_result) - np.min(lic_result))
         # Step 4: Convert to an image and save
        lic_normalized_img = (lic_normalized * 255).astype(np.uint8)  # Convert to 8-bit grayscale
        img = Image.fromarray(lic_normalized_img, mode='L')
        save_name=f"{saveName}_{i}.png"
        savePath=os.path.join(saveFolder,save_name)
        img.save(savePath)



@typechecked
def LicRenderingSteadyCpp(vecfield: SteadyVectorField2D,licImageSizeX:int,licImageSizeY:int,saveFolder:str="./",saveName:str="vector_field_lic",stepSize=0.01, MaxIntegrationSteps=128):
    """
    Render a steady 2D vector field as an LIC image and save to a PNG file.
    """ 

    assert(cppMoudules['CppLicRenderingModule'].licRenderingPybindCPP is not None)

    lic_result = cppMoudules['CppLicRenderingModule'].licRenderingPybindCPPv2( vecfield.field, vecfield.Xdim, vecfield.Ydim, vecfield.domainMinBoundary[0], vecfield.domainMaxBoundary[0], vecfield.domainMinBoundary[1], vecfield.domainMaxBoundary[1],licImageSizeX,licImageSizeY,stepSize,MaxIntegrationSteps)

    lic_normalized = np.clip(lic_result, 0, np.max(lic_result))  # Clip negative values to 0
    # Step 4: Convert to an image and save
    lic_normalized_img = (lic_normalized * 255).astype(np.uint8)  # Convert to 8-bit grayscale
    img = Image.fromarray(lic_normalized_img, mode='RGB')

    save_name=saveName if saveName.endswith("png") else f"{saveName}.png"
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    savePath=os.path.join(saveFolder,save_name)
    img.save(savePath)
    return img

  


@typechecked
def LicRenderingPathlineSegmentation(vecfield: UnsteadyVectorField2D, SegmentationImage, upsampling: float = 2.0, saveFolder: str = "./", saveName: str = "vector_field_lic", stepSize = 0.01, MaxIntegrationSteps = 128):
    """
    Render a LIC image with pathline segmentation overlay.
    """
    # Convert segmentation image to numpy array
    segmentation_array = np.array(SegmentationImage)
    
    # Create colored segmentation image
    segmentation_colored = np.zeros((*segmentation_array.shape, 3), dtype=np.uint8)
    segmentation_colored[segmentation_array >= 0.6] = [255, 255, 0]  # Yellow for class 1
    
    # Apply upsampling if needed
    if upsampling != 1.0:
        new_size = (int(segmentation_array.shape[0] * upsampling), int(segmentation_array.shape[1]* upsampling))
        segmentation_colored = Image.fromarray(segmentation_colored).resize(new_size, Image.NEAREST)
        segmentation_colored = np.array(segmentation_colored)
    else:
        new_size = SegmentationImage.size

    # Generate LIC image
    assert cppMoudules['CppLicRenderingModule'].licRenderingPybindCPP is not None
    steadyVectorField2D = vecfield.getSlice(0)
    lic_result = cppMoudules['CppLicRenderingModule'].licRenderingPybindCPP(
        steadyVectorField2D.field, vecfield.Xdim, vecfield.Ydim,
        vecfield.domainMinBoundary[0], vecfield.domainMaxBoundary[0],
        vecfield.domainMinBoundary[1], vecfield.domainMaxBoundary[1],
        new_size[0], stepSize, MaxIntegrationSteps
    )
    
    # Normalize and convert LIC result to RGB
    lic_normalized = np.clip(lic_result, 0, np.max(lic_result))  # Clip negative values to 0
    # Step 4: Convert to an image and save
    lic_color_img = (lic_normalized * 255).astype(np.uint8)  # Convert to 8-bit grayscale

    # Blend LIC image with segmentation
    blended_image = np.clip(lic_color_img * 0.5 + segmentation_colored * 0.5, 0, 255).astype(np.uint8)

    # Save images
    os.makedirs(saveFolder, exist_ok=True)
    
    blended_image = Image.fromarray(blended_image, mode='RGB')
    blended_save_path = os.path.join(saveFolder, f"{saveName}.png")
    blended_image.save(blended_save_path)
    return blended_image

@typechecked
def LicRenderingUnsteadyCpp(vecfield:UnsteadyVectorField2D,licImageSizeX:int,licImageSizeY:int,timeStepSKip:int=2,saveFolder:str="./",saveName:str="vector_field_lic",stepSize=0.01, MaxIntegrationSteps=128):
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    #typecheck field type and field is not None    
    Xdim,Ydim,time_steps=vecfield.Xdim,vecfield.Ydim,vecfield.time_steps
    for i in range(0, time_steps, timeStepSKip):
        # print(f"Processing time step {i}")
        steadyVectorField2D = vecfield.getSlice(i)
        save_name=f"{saveName}_{i}"
        LicRenderingSteadyCpp(steadyVectorField2D ,licImageSizeX,licImageSizeY, saveFolder=saveFolder,saveName =save_name,stepSize=stepSize,MaxIntegrationSteps=MaxIntegrationSteps)
    
    
    
@typechecked        
def LicGlyphMixRenderingSteady(vecfield: SteadyVectorField2D,licImageSize:int,saveFolder:str="./",saveName:str="vector_field_lic",stepSize=0.01, MaxIntegrationSteps=128,ColorCodingFn=lambda u, v: math.sqrt(u*u + v*v)):
    assert(cppMoudules['CppLicRenderingModule'].licRenderingPybindCPP is not None)
    lic_normalized_array = cppMoudules['CppLicRenderingModule'].licRenderingPybindCPP( vecfield.field, vecfield.Xdim, vecfield.Ydim, vecfield.domainMinBoundary[0], vecfield.domainMaxBoundary[0], vecfield.domainMinBoundary[1], vecfield.domainMaxBoundary[1],licImageSize,stepSize,MaxIntegrationSteps)
    lic_normalized_array=(np.clip(lic_normalized_array, 0, np.max(lic_normalized_array)) *255).astype(np.uint8)
    glyImage:Image.Image=glyphsRenderSteadyFieldAlgorthim(vecfield,(licImageSize,licImageSize),ColorCodingFn=ColorCodingFn,gridSkip=1)
    glyImage_array = np.array(glyImage)
    white_mask = np.all(glyImage_array == [255, 255, 255], axis=-1)
    mix_image_array = np.where(white_mask[..., None], lic_normalized_array, glyImage_array)
    mix_image:Image.Image =  Image.fromarray(mix_image_array,mode='RGB') # Clip negative values to 0
    save_name=saveName if saveName.endswith("png") else f"{saveName}.png"
    savePath=os.path.join(saveFolder,save_name)
    mix_image.save(savePath)

@typechecked
def LicGlyphMixRenderingUnsteady(vecfield:UnsteadyVectorField2D,licImageSize:int,timeStepSKip:int=2,saveFolder:str="./",saveName:str="vector_field_lic",stepSize=0.01, MaxIntegrationSteps=128, ColorCodingFn=lambda u, v: math.sqrt(u*u + v*v)):
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    #typecheck field type and field is not None    
    Xdim,Ydim,time_steps=vecfield.Xdim,vecfield.Ydim,vecfield.time_steps
    texture = np.random.rand(Xdim, Ydim)    
    for i in range(0, time_steps, timeStepSKip):
        # print(f"Processing time step {i}")
        steadyVectorField2D = vecfield.getSlice(i)
        f_save_name=f"{saveName}_{i}"
        LicGlyphMixRenderingSteady(steadyVectorField2D ,licImageSize, saveFolder=saveFolder,saveName =f_save_name,stepSize=stepSize,MaxIntegrationSteps=MaxIntegrationSteps,ColorCodingFn=ColorCodingFn)


