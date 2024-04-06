#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
uniform mat4 projMat;
uniform mat4 viewMat;
uniform mat4 modelMat;
out vec3 pos;

void main()
{	
	gl_Position = projMat * (viewMat * modelMat) * vec4(aPos, 1);
	// TexCoords=aTexCoord;
	pos = aPos;
}
