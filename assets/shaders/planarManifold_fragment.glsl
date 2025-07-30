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
    float normalizedTime = (time - scalarFieldMinTime) / (scalarFieldMaxTime - scalarFieldMinTime);
    normalizedTime=clamp(normalizedTime,0.0,1.0);
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
