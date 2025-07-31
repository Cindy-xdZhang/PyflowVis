#version 460 core
layout (location = 0) in vec3 aPos;
uniform mat4 projMat;
uniform mat4 viewMat;
uniform mat4 modelMat;
uniform float uplifting;
uniform float zOffset;
out vec3 vPos;
uniform int instanceType;//0:curve_2D,1:curve_3D

vec3 uplifting_curve_2D(inout vec3 pos){
	vec3 n = vec3(0, 0, 1);
	vec3 uplifting_vec = n * (aPos.z * uplifting+zOffset);
	return pos+uplifting_vec;
}





void main()
{	
	vec3 resultpos =aPos;
	if(instanceType==0){
		resultpos=uplifting_curve_2D(resultpos);
	}


	gl_Position = projMat * (viewMat * modelMat) * vec4(resultpos, 1);
	// TexCoords=aTexCoord;
	vPos=resultpos;
}
