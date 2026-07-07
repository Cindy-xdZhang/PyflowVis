#version 460 core
// Direct volume rendering by object-space ray marching, ported from the optimal-connection
// engine (src/shaders/DVR.frag / DVR_OS). Single pass over the proxy cube's back faces; the
// ray/box segment is found analytically (slab method) so entry/exit do not depend on which
// face rasterized. Feature parity with the C++ DVR:
//   - view-independent step size tied to the voxel grid (uSampleStepFactor, uMaxSteps cap);
//   - per-pixel jitter of the entry point (removes wood-grain banding);
//   - step-size opacity correction with density as a real extinction coefficient (uDensityScale);
//   - central-difference gradient Blinn-Phong shading (uShadingMode, 0..3);
//   - min-max brick empty-space skipping (uMinMaxTex, uEnableEmptySkip).
// Two intentional adaptations to the Python engine (which has no scene-light uniforms):
//   - the light is a headlight (co-located with the camera), needing only camPos;
//   - output is premultiplied alpha, composited with glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA).
in vec3 vWorldPos;

uniform mat4 modelMat;
uniform vec3 camPos;                  // world-space camera position (uniform, not per-fragment inverse)
uniform vec3 uVolumeBoundsMin;        // object-space AABB of the volume
uniform vec3 uVolumeBoundsMax;
uniform sampler3D uVolumeTex;         // R16F raw scalar volume
uniform sampler1D uTransferFunc;      // RGBA transfer-function LUT
uniform vec2 attributeBounds;         // [min,max] of the scalar data, for normalization
uniform float lightStrength;

uniform int   uMaxSteps;              // safety cap on the step count
uniform float uDensityScale;          // extinction coefficient (thickens/dims the whole volume)
uniform float uSampleStepFactor;      // base step as a fraction of the smallest voxel (e.g. 0.5)
uniform int   uShadingMode;           // 0: none; 1/2/3: box-smooth + gradient shading
uniform sampler3D uMinMaxTex;         // coarse per-brick (min,max) density for empty-space skipping
uniform int   uEnableEmptySkip;       // 0/1

// Depth-aware compositing: clamp the ray to the opaque scene surface so geometry in front of the
// volume occludes it, and geometry intersecting the box no longer truncates the whole ray.
uniform sampler2D uSceneDepth;        // copy of the opaque depth buffer
uniform vec2  uViewport;              // (width, height) in pixels
uniform mat4  uInvViewProj;           // inverse(projMat * viewMat), to unproject scene depth
uniform int   uUseSceneDepth;         // 0/1

out vec4 FragColor;

// Ray/AABB intersection (slab method); rayDir components of 0 give +/-inf and still work.
bool intersectBox(in vec3 rayOrigin, in vec3 rayDir, out float tEnter, out float tExit)
{
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (uVolumeBoundsMin - rayOrigin) * invDir;
    vec3 t1 = (uVolumeBoundsMax - rayOrigin) * invDir;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    tEnter = max(max(tmin.x, tmin.y), tmin.z);
    tExit  = min(min(tmax.x, tmax.y), tmax.z);
    return tExit > tEnter;
}

// Cheap hash -> [0,1) for per-pixel ray-start jitter (Dave Hoskins' hash12).
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec4 DVR_OS()
{
    vec3 rayOriginWS = camPos;
    vec3 rayDirWS    = normalize(vWorldPos - camPos);

    // Move the ray into object space so the AABB test uses uVolumeBoundsMin/Max directly.
    mat4 invModel    = inverse(modelMat);
    vec3 rayOriginOS = (invModel * vec4(rayOriginWS, 1.0)).xyz;
    vec3 p1OS        = (invModel * vec4(rayOriginWS + rayDirWS, 1.0)).xyz;
    vec3 rayDirOS    = normalize(p1OS - rayOriginOS);

    float tEnter, tExit;
    if (!intersectBox(rayOriginOS, rayDirOS, tEnter, tExit))
        discard;
    tEnter = max(tEnter, 0.0);

    // Depth-aware occlusion: stop the ray at the nearest opaque surface along it. Volume in front
    // of geometry still shows; volume behind it is skipped — instead of the whole ray being culled.
    // NOTE: tEnter/tExit are OBJECT-space parameters (from the OS box test above), so the scene hit
    // must be parameterized in object space too. Transform the reconstructed world hit back through
    // invModel and project onto rayDirOS. Doing the clamp in object space keeps it correct even if
    // the volume's modelMat is ever non-identity/scaled — do NOT compare a world-space t to tExit.
    if (uUseSceneDepth != 0) {
        vec2 suv = gl_FragCoord.xy / uViewport;
        float dz = texture(uSceneDepth, suv).r;
        if (dz < 1.0) {
            vec4 ndc = vec4(suv * 2.0 - 1.0, dz * 2.0 - 1.0, 1.0);
            vec4 wp = uInvViewProj * ndc;
            wp /= wp.w;
            vec3 wpOS = (invModel * vec4(wp.xyz, 1.0)).xyz;
            float tScene = dot(wpOS - rayOriginOS, rayDirOS);
            tExit = min(tExit, tScene);
        }
    }

    if (tEnter >= tExit)
        discard;

    float maxDistance = tExit - tEnter;
    float used_lightStrength = lightStrength * 0.5;

    vec3 volumeSize = vec3(textureSize(uVolumeTex, 0));
    vec3 texelSize  = 1.0 / volumeSize;

    // Fixed sampling step tied to the voxel grid (object space): sampling rate is independent of
    // view angle and of uMaxSteps, which is now only a safety cap. refStep is the reference step
    // at which the transfer-function alpha is authored.
    vec3  voxelSizeOS = (uVolumeBoundsMax - uVolumeBoundsMin) * texelSize;
    float refStep  = max(min(voxelSizeOS.x, min(voxelSizeOS.y, voxelSizeOS.z)) * uSampleStepFactor, 1e-6);
    int   maxSteps = max(uMaxSteps, 1);
    float stepSize = max(refStep, maxDistance / float(maxSteps));

    // Per-pixel jitter of the entry point breaks the regular sampling planes into noise the eye ignores.
    float jitter = hash12(gl_FragCoord.xy);

    vec4  accum = vec4(0.0);
    float t = tEnter + jitter * stepSize;

    // Empty-space skipping bookkeeping: brick size (object space) and a "current brick" tracker so the
    // costly per-brick transfer-function test runs only when the ray crosses into a new brick.
    ivec3 brickDims   = max(textureSize(uMinMaxTex, 0), ivec3(1));
    vec3  brickSizeOS = (uVolumeBoundsMax - uVolumeBoundsMin) / vec3(brickDims);
    ivec3 lastBrick   = ivec3(-2147483647);

    for (int i = 0; i < maxSteps; ++i) {
        if (t > tExit) break;
        if (accum.a > 0.95) break;      // early ray termination

        vec3 pos = rayOriginOS + t * rayDirOS;
        vec3 texCoord = (pos - uVolumeBoundsMin) / (uVolumeBoundsMax - uVolumeBoundsMin);

        // Empty-space skipping: on entering a new brick, if the transfer function assigns no opacity
        // anywhere in this brick's [min,max] density range, jump to the brick's far boundary. The brick
        // volume is built with a 1-voxel halo so trilinear sampling at brick edges stays covered.
        if (uEnableEmptySkip != 0) {
            ivec3 curBrick = ivec3(floor((pos - uVolumeBoundsMin) / brickSizeOS));
            if (curBrick != lastBrick) {
                lastBrick = curBrick;
                vec2 mm = texelFetch(uMinMaxTex, clamp(curBrick, ivec3(0), brickDims - 1), 0).rg;
                float sMin = clamp((mm.x - attributeBounds.x) / (attributeBounds.y - attributeBounds.x), 0.0, 1.0);
                float sMax = clamp((mm.y - attributeBounds.x) / (attributeBounds.y - attributeBounds.x), 0.0, 1.0);
                float maxA = 0.0;
                for (int k = 0; k <= 16; ++k)
                    maxA = max(maxA, texture(uTransferFunc, mix(sMin, sMax, float(k) / 16.0)).a);
                if (maxA < 0.004) {
                    vec3 farFace = (vec3(curBrick) + step(0.0, rayDirOS)) * brickSizeOS + uVolumeBoundsMin;
                    vec3 tNext   = (farFace - pos) / rayDirOS; // >0 per axis; +inf for zero-dir axes
                    t += min(tNext.x, min(tNext.y, tNext.z)) + refStep * 0.5;
                    continue;
                }
            }
        }

        float density = 0.0;
        int shadingMode = uShadingMode;
        if (shadingMode == 0) {
            density = texture(uVolumeTex, texCoord).r;
        } else {
            // Box-smooth the density over a (2*offset+1)^3 window; the shading normal below is a
            // proper central-difference gradient, not this accumulation.
            float totalWeight = 0.0;
            float startCoordOffset = 0.5 + float(shadingMode) * 0.5;
            for (float z = -startCoordOffset; z <= startCoordOffset; z += 1.0) {
                for (float y = -startCoordOffset; y <= startCoordOffset; y += 1.0) {
                    for (float x = -startCoordOffset; x <= startCoordOffset; x += 1.0) {
                        density += texture(uVolumeTex, texCoord + vec3(x, y, z) * texelSize).r;
                        totalWeight += 1.0;
                    }
                }
            }
            density /= totalWeight;
        }

        // Classify by the raw density normalized to the data range. uDensityScale does NOT distort
        // this lookup; it is applied below as a real extinction coefficient.
        float scalar = clamp((density - attributeBounds.x) / (attributeBounds.y - attributeBounds.x), 0.0, 1.0);
        vec4 col = texture(uTransferFunc, scalar);

        if (shadingMode > 0 && col.a > 0.01) {
            // Central-difference gradient in PHYSICAL (object-space) units: dividing by the per-axis
            // voxel size keeps the normal correct on anisotropic / non-cube grids.
            float dxp = texture(uVolumeTex, texCoord + vec3(texelSize.x, 0.0, 0.0)).r;
            float dxm = texture(uVolumeTex, texCoord - vec3(texelSize.x, 0.0, 0.0)).r;
            float dyp = texture(uVolumeTex, texCoord + vec3(0.0, texelSize.y, 0.0)).r;
            float dym = texture(uVolumeTex, texCoord - vec3(0.0, texelSize.y, 0.0)).r;
            float dzp = texture(uVolumeTex, texCoord + vec3(0.0, 0.0, texelSize.z)).r;
            float dzm = texture(uVolumeTex, texCoord - vec3(0.0, 0.0, texelSize.z)).r;
            vec3 gradOS = vec3(dxp - dxm, dyp - dym, dzp - dzm) / (2.0 * voxelSizeOS);

            if (length(gradOS) > 1e-6) {
                // Lighting in WORLD space; the normal transforms by inverse-transpose(model).
                vec3 posWS     = (modelMat * vec4(pos, 1.0)).xyz;
                mat3 normalMat = transpose(mat3(invModel));
                vec3 N = normalize(normalMat * (-gradOS));  // outward surface normal (-grad density)
                vec3 V = normalize(camPos - posWS);
                if (dot(N, V) < 0.0) N = -N;                // two-sided: always light the visible face
                vec3 L = V;                                  // headlight (no scene light in this engine)
                vec3 H = normalize(L + V);
                float diff = max(dot(N, L), 0.0);
                float spec = pow(max(dot(N, H), 0.0), 32.0);
                const float ambient = 0.3;
                col.rgb = col.rgb * (ambient + 0.8 * diff * used_lightStrength) + vec3(0.25 * spec * used_lightStrength);
            }
        }

        // Step-size opacity correction with density as a real extinction multiplier: the TF alpha is
        // the base opacity for one refStep; optical depth scales with the actual stepSize and with
        // uDensityScale, so uDensityScale physically thickens/dims the volume.
        float alpha = 1.0 - pow(max(1.0 - col.a, 1e-6), uDensityScale * stepSize / refStep);

        float oneMinusA = 1.0 - accum.a;
        accum.rgb += oneMinusA * col.rgb * alpha;   // front-to-back, premultiplied
        accum.a   += oneMinusA * alpha;

        t += stepSize;
    }

    if (accum.a < 0.01)
        discard;
    return accum;   // premultiplied alpha
}

void main()
{
    FragColor = DVR_OS();
}
