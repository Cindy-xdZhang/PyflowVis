#version 460 core
out vec4 FragColor;

in vec3  vPos;
in vec2 texUV;


uniform  float time;
uniform float scalarFieldMinTime;
uniform float scalarFieldMaxTime;


// color map for scalar attribute
uniform sampler3D scalarAttributeTexture;
uniform vec2 attributeBounds;
uniform sampler1DArray colorMaps1Darray;
uniform  int colorMap;


vec4 getScalarFieldValueOnPlane(vec2 texUV, float time) {
    if (time > scalarFieldMaxTime)
        return vec4(0, 0, 0, 0);
    if (time < scalarFieldMinTime)
        return vec4(0, 0, 0, 0);
    float normalizedTime = (time - scalarFieldMinTime) / (scalarFieldMaxTime - scalarFieldMinTime);
    vec3 attributeTexCoords = vec3(texUV, normalizedTime);
    return texture(scalarAttributeTexture, attributeTexCoords);
}


void main()
{
	

    float scalarValue=getScalarFieldValueOnPlane(texUV,time).x;
    float normalized_scalarValue=(scalarValue-attributeBounds.x)/(attributeBounds.y-attributeBounds.x);
    vec2 final_texArrayCoords = vec2(normalized_scalarValue, colorMap);
    vec4 textureColor = texture(colorMaps1Darray, final_texArrayCoords);
    FragColor = textureColor;
    // FragColor = vec4(1,scalarValue,0,1);
}
