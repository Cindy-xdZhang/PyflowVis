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



//-----------------------------------------------------------//--------------------------------------------------------------------
//-----------------------------------------------------------//--------------------------------------------------------------------
//-----------------------------------------------------------//--------------------------------------------------------------------
//-----------------------------------------------------------//--------------------------------------------------------------------
//------------------------------------------observer relavive transforamtion funcionality------------------------------------------
//-----------------------------------------------------------//--------------------------------------------------------------------
//-----------------------------------------------------------//--------------------------------------------------------------------
//-----------------------------------------------------------//--------------------------------------------------------------------

layout(std430, binding = 6) buffer SSBOBufferId0
{
    vec4 worldlineAndIntegratedC[];
}
ssboBufferObject0;
uniform int ssbo0_length;
uniform float DefaultObseverMaxTime;
uniform float DefaultObseverMinTime;
uniform float InputFieldMaxTime;
uniform float InputFieldMinTime;
uniform vec2 worldlineStartPos0;

vec3 getTransformationDefault(float normalized_time)
{
    float CorrespondingWorldtime = normalized_time * (InputFieldMaxTime - InputFieldMinTime) + InputFieldMinTime;
    float ObseverMinTime = DefaultObseverMinTime;
    float ObseverMaxTime = DefaultObseverMaxTime;

    int numberOfElements = ssbo0_length;
    
    vec3 res;
    if (CorrespondingWorldtime >= ObseverMinTime && CorrespondingWorldtime <= ObseverMaxTime) {
        float NomarlizedTimeInObserverTimeRange = (CorrespondingWorldtime - ObseverMinTime) / (ObseverMaxTime - ObseverMinTime);
        float fIdxf = floor(NomarlizedTimeInObserverTimeRange * (numberOfElements - 1));
        float fIdxc = min(fIdxf + 1.0, numberOfElements - 1); // maximum is (numberOfElements - 1)

        int iIdxf = int(fIdxf);
        int iIdxc = int(fIdxc);

        float alpha = NomarlizedTimeInObserverTimeRange* (numberOfElements - 1.0) - fIdxf;
        vec3 c_floor = ssboBufferObject0.worldlineAndIntegratedC[iIdxf].xyz;
        vec3 c_ceil = ssboBufferObject0.worldlineAndIntegratedC[iIdxc].xyz;
        
        vec3 interpolatedValue = c_floor * (1.f - alpha) + c_ceil * alpha;
        res = interpolatedValue;
    }
    else if (CorrespondingWorldtime < ObseverMinTime) {
        res = ssboBufferObject0.worldlineAndIntegratedC[0].xyz;
    }
    else {
        int idx;
        idx = int(numberOfElements - 1);
        res = ssboBufferObject0.worldlineAndIntegratedC[idx].xyz;
    }
    return res;
}


mat3 getObserverTransformation(float normalized_time)
{
    vec3 trans0 = getTransformationDefault(normalized_time);

    vec3 res = trans0 ;
    vec2 observer_pos = res.xy;
    float interpolatedC = res.z;
    vec2 worldlineStartPos = worldlineStartPos0;
    float start_px = worldlineStartPos.x;
    float start_py = worldlineStartPos.y;

    mat3 transMatP1 = mat3(
        vec3(1, 0, 0),
        vec3(0, 1, 0),
        vec3(-observer_pos.x, -observer_pos.y, 1));
        

    mat3 transMatP2 = mat3(
        vec3(1, 0, 0),
        vec3(0, 1, 0),
        vec3(start_px, start_py, 1));

    float sinTheta = sin(interpolatedC);
    float cosTheta = cos(interpolatedC);

    // the constructor of mat3 is col-major
    mat3 rotateMat = mat3(
        vec3(cosTheta, sinTheta, 0),
        vec3(-sinTheta, cosTheta, 0),
        vec3(0, 0, 1));
    mat3 rotateInverse = transpose(rotateMat);

    mat3 inverseTransformation = (transMatP2 * rotateInverse * transMatP1);
    return inverseTransformation;
}

vec2 referenceFrameTransform(vec2 p, float normalizedTime)
{
      mat3   transformation ;
      if (transformationMode == 1) {
         transformation = getTransformationDefault(normalizedTime);
      }
      else {
         return p;
      }
    vec3 pos = vec3(p,1);
    pos = transformation * pos;
    return pos.xy;
}



void main() {

     //observer transforamtion+uplifting
    //vec3 resultpos = vec3(transformedInSpace, 0) + n;

    //uplifting
    vec3 n = vec3(0, 0, 1) * (in_attribs.x * uplifting);
    n += vec3(0, 0, 1) * zOffset;


    vec2 pos2d = in_position.xy;
    //observer transforamtion
    vec2 transformedInSpace = referenceFrameTransform(pos2d, in_attribs.x);
    vec3 resultpos = vec3(transformedInSpace, 0) + n;

    
    gl_Position = viewMat * modelMat * vec4(resultpos, 1);
    vs_out.position = resultpos;

    vs_out.attrib = in_attribs.x;
    vs_out.attrib2 = in_attribs.y;

    //opcaity control ignore for now.
    float animation_opacity = 1.0;
    vs_out.opacity = animation_opacity ;
}