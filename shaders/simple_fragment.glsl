#version 460 core
out vec4 FragColor;

in vec2 TexCoords;

//uniform sampler2D texture1;

void main()
{
     //produce texture 
     float stripeWidth = 0.05; 
  	float gapWidth = 0.05; 
     float pattern = step(stripeWidth, mod(TexCoords.x + gapWidth, stripeWidth + gapWidth));
     vec3 color = mix(vec3(1.0), vec3(0.0), pattern);
     FragColor = vec4(color, 1.0);
}
