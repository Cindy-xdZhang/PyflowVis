#version 460 core
in vec3 position;
in vec3 tangent;
in float attrib;
in float attrib2;
in float opacity;

uniform int colorMap;
uniform sampler1DArray colorMaps1Darray;
out vec4 FragColor;

uniform vec3 light;
float diffuse(vec3 T, vec3 L)
{
   float kd = 0.7;
   float diff = sqrt( 1 - pow(dot(L, T), 2));
   return kd * max(diff, 0.0);
}

vec4 getLineGroupColorMapTexure(float normalized_attribute){
    int selected_colorMap_id;
 //if(flowlineGroupID==SHADER_SHARE_MACRO_FLOWLINE_PATHLINE_2D )//pathline 2D
//{
 //   selected_colorMap_id=colorMap;
//}
//if(flowlineGroupID==SHADER_SHARE_MACRO_FLOWLINE_STREAMLINE_3D )//streamline 3D
//{

//}

    selected_colorMap_id=colorMap;
    vec2 texArrayCoords = vec2(normalized_attribute, selected_colorMap_id);
    return  texture(colorMaps1Darray, texArrayCoords);

}

void main() {
    if(opacity <= 0.01)
    {
        discard;
    } 
    vec3 lightDir = normalize(light - position);
    vec3 T = normalize(tangent);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    float ambient = 0.05;
    float diffuse = diffuse(T, lightDir);

    FragColor = getLineGroupColorMapTexure(attrib);
    FragColor.w = opacity;
}