#version 460 core
out vec4 FragColor;

in vec3  vPos;

// color map for scalar attribute
uniform sampler1DArray colorMaps1Darray;
uniform  int colorMap;
uniform vec2 attributeBounds;

void main()
{
	float attrib= vPos.x;
    float scaledAttrib = (attrib - attributeBounds.x) / (attributeBounds.y - attributeBounds.x);
    vec2 texArrayCoords = vec2(scaledAttrib, colorMap);
    FragColor = texture(colorMaps1Darray, texArrayCoords);

    
}
