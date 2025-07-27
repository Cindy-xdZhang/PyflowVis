#version 460 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_attribs;


uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

uniform float uplifting;
uniform float zOffset;

out VS_OUT {
    vec3 position;
    float attrib;
    float attrib2;
    float opacity;
} vs_out;

void main() {

     //observer transforamtion+uplifting
    //vec3 resultpos = vec3(transformedInSpace, 0) + n;

    //uplifting
    vec3 n = vec3(0, 0, 1) * (in_attribs.x * uplifting);
    n += vec3(0, 0, 1) * zOffset;
    vec3 resultpos =in_position + n;


    
    gl_Position = viewMat * modelMat * vec4(resultpos, 1);
    vs_out.position = resultpos;

    vs_out.attrib = in_attribs.x;
    vs_out.attrib2 = in_attribs.y;

    //opcaity control ignore for now.
    float animation_opacity = 1.0;
    vs_out.opacity = animation_opacity ;
}