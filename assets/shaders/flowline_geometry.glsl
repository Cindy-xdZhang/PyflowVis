#version 460 core


// #shader


layout (lines_adjacency) in;
layout (triangle_strip, max_vertices=4) out;


uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

uniform float lineWeight;
uniform float zOffset;
uniform unsigned int flowlineGroupID; 
//layout (triangle_strip, max_vertices=4) out;

in VS_OUT {
    vec3 position;
    //vec3 tangent;
    float attrib;
     float attrib2;
    float opacity;
} gs_in[];


out vec3 position;
out vec3 tangent;
out float attrib;
out float attrib2;
out float opacity;

void main() {    

    //vec3 direction = gs_in[1].position.xyz - gs_in[0].position.xyz;
    //direction = normalize(direction);
    float use_line_wight=lineWeight;
    if(flowlineGroupID==1)
        use_line_wight=0.25*use_line_wight;
    
    vec4 previous = gl_in[0].gl_Position; 
    vec4 start = gl_in[1].gl_Position;
    vec4 end = gl_in[2].gl_Position;
    vec4 next = gl_in[3].gl_Position;

    vec3 directionPrevious = start.xyz-previous.xyz;
    vec3 directionThis = end.xyz-start.xyz;
    vec3 directionNext = next.xyz-end.xyz;
    
    //vec3 endBisectorDirection = normalize(normalize(directionThis) - normalize(directionNext));
    //vec3 startBisectorDirection = normalize(normalize(directionPrevious) - normalize(directionThis));
    
    vec3 endTangent = normalize((directionThis + directionNext) * 0.5f);
    vec3 startTangent = normalize((directionPrevious + directionThis) * 0.5f);

    vec3 viewingDirection = vec3(0,0,-1);

    vec3 startOffset = cross(startTangent, viewingDirection*(1.0/start.z));
    vec3 endOffset = cross(endTangent, viewingDirection*(1.0/end.z));



    position = gs_in[1].position;
    tangent = startTangent;
    attrib = gs_in[1].attrib;
    opacity = gs_in[1].opacity;
    attrib2=gs_in[1].attrib2;

    gl_Position = projMat*(vec4(0,0,0,0) + start - vec4(startOffset*use_line_wight,0));
    EmitVertex();
    gl_Position = projMat*(vec4(0,0,0,0) + start + vec4(startOffset*use_line_wight,0));
    EmitVertex();

    position = gs_in[2].position;
    tangent = endTangent;
    attrib = gs_in[2].attrib;
    opacity = gs_in[2].opacity;
    attrib2=gs_in[2].attrib2;

    gl_Position = projMat*(vec4(0,0,0,0) + end - vec4(endOffset*use_line_wight,0));
    EmitVertex();
    gl_Position = projMat*(vec4(0,0,0,0) + end + vec4(endOffset*use_line_wight,0));
    EmitVertex();
    EndPrimitive();
    
    


}  