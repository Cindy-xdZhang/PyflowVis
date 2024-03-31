from     Object import *


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
  