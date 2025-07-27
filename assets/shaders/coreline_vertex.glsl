#version 460 core
layout (location = 0) in vec3 aPos;
uniform mat4 projMat;
uniform mat4 viewMat;
uniform mat4 modelMat;
uniform float uplifting;
uniform float zOffset;
out vec3 vPos;

void main()
{	
	vec3 n = vec3(0, 0, 1) * (aPos.z * uplifting);
    n += vec3(0, 0, 1) * zOffset;
	vec3 resultpos =aPos;
	resultpos.z=n.z;

	gl_Position = projMat * (viewMat * modelMat) * vec4(resultpos, 1);
	// TexCoords=aTexCoord;
	vPos=resultpos;
}
