#include "brdf.hlsl"
#include "tracing.hlsl"
#include "sampling.hlsl"

Texture2D<float> g_blueNoise : register(t0);
RWTexture2D<float4> g_output : register(u0);
RWByteAddressBuffer g_brdfBuff : register(u1);

#ifndef SAMPLE_LINES
#define SAMPLE_LINES 64
#endif

cbuffer constants : register(b0)
{
    float4 g_sizes;
    float4 g_angles;
    float4 g_material;
    float4 g_coordTransform;
    float4 g_eyeArgs0;
    float4 g_eyeArgs1;
    float4 g_lightArgs0;
    float4x4 g_lightTransform;
}

#define g_VdotN g_angles.x
#define g_roughness g_material.x
#define g_eyePos g_eyeArgs0.xyz
#define g_eyeAzimuth g_eyeArgs0.w
#define g_eyeAltitude g_eyeArgs1.x
#define g_eyeCosFovY g_eyeArgs1.y
#define g_eyeSinFovY g_eyeArgs1.z
#define g_lightIntensity g_lightArgs0.x
#define g_lightWidth g_lightArgs0.y
#define g_lightHeight g_lightArgs0.z
#define g_lightSamples (uint)asint(g_lightArgs0.w)

float3 drawLine(float2 p0, float2 p1, float thickness, float3 col, float2 coord)
{
    float2 dVec = p1 - p0;
    float dist = length(dVec);
    float2 d = dVec/dist;
    float proj = dot(d, coord - p0);
    float2 q = p0 + d * proj;
    float t = saturate(1.0 - pow(distance(q, coord) / thickness, 4.0));
    t = (proj < 0.0 || proj > dist) ? 0.0 : t;
    return saturate(col * t);
}
 
float2 hemisphereSample2D(uint i, uint count)
{
    float ang = ((float)i/(float)count) * PI;
    return float2(cos(ang), sin(ang));
} 

float3 drawBrdf(float2 o, float lineThickness, float2 V, float2 N, float roughness, float2 coord)
{
    uint i = 0;
    float3 col = float3(0,0,0);

    float NdotV = dot(N, V);
    float lambda = GetSmithJointGGXPartLambdaV(NdotV, roughness);
    for (i = 0; i <= SAMPLE_LINES; ++i)
    {
        float2 L = hemisphereSample2D(i, SAMPLE_LINES);
        float2 H = normalize(L + V);
        float NdotL = dot(N, L);
        float NdotH = dot(N, H);
        float VdotH = dot(V, H);

        float vis = DV_SmithJointGGX(NdotH, NdotL, max(NdotV,0), roughness, lambda);
        float weightOverPdf = 4.0 * vis * NdotL * VdotH / NdotH;
        float FweightOverPdf = weightOverPdf;

        float ww = weightOverPdf * NdotL;

        col += drawLine(o, o + ww * L, lineThickness, float3(0.0, 0.0, 0.4), coord);
    }
    return col;
}

float integrateBrdf(float3 V, float roughness)
{
    float3 N = float3(0,1,0);
    float intValue = 0.0;
    float NdotV = dot(N, V);
    float lambda = GetSmithJointGGXPartLambdaV(NdotV, roughness);
    uint i;
    for (i = 0; i < g_lightSamples; ++i)
    {
        float2 uv = sampleHammersley(i, g_lightSamples);
        float3 L = sampleCosineHemisphere(uv.x, uv.y).xzy;
        float3 H = normalize(L + V);
        float NdotL = dot(N, L);
        float NdotH = dot(N, H);
        float VdotH = dot(N, H);

        float vis = DV_SmithJointGGX(NdotH, NdotL, max(NdotV,0), roughness, lambda);
        float weightOverPdf = 4.0 * vis * NdotL * VdotH / NdotH;
        float FweightOverPdf = weightOverPdf * F_Schlick(0.5, NdotV);

        float ww = FweightOverPdf;
        intValue += ww;
    }

    intValue *= 1.0 / (float)g_lightSamples;
    return intValue;
}

float2 getHCoords(uint2 pixelCoord)
{
    float2 uv = (pixelCoord + 0.5) * g_sizes.zw;
    float2 coord = uv * 2.0 - 1.0;
    coord.y *= -1;
    return coord;
}

float2 getCanvasCoord(uint2 pixelCoord)
{
    float2 coord = getHCoords(pixelCoord);
    float aspectInv = g_sizes.x/g_sizes.y;
    coord.x *= aspectInv;

    coord.xy += g_coordTransform.xy;
    coord.xy /= g_coordTransform.z;
    return coord;
}

[numthreads(8,8,1)]
void csBrdf2DPreview(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float2 coord = getCanvasCoord(dispatchThreadID.xy);

    float2 N = float2(0.0, 1.0);
    float2 o = float2(0, -0.3);
    float3 col = float3(0,0,0);
    float2 V = float2(sqrt(max(1.0 - g_VdotN*g_VdotN, 0)),g_VdotN);

    float2 Vi = 1.7 * V * float2(-1,1);
    float2 Vo = 1.7 * V * float2( 1,1);
    float lineThickness = 0.004 / g_coordTransform.z;
    col += drawLine(float2(-4,o.y), float2(4,o.y), lineThickness, float3(0.4,0,0), coord);
    col += drawLine(float2(o.x,-1), float2(o.x,1), lineThickness, float3(0.2,0.2,0.2), coord);
    col += drawLine(o, o + Vo, lineThickness, float3(0.2,0.2,0.2), coord);

    float brdfV;
    col += drawBrdf(o, lineThickness, float2(V.x, V.y), N, g_roughness, coord);

    if (dispatchThreadID.x == 0 && dispatchThreadID.y == 0)
        g_brdfBuff.Store(0, asuint(integrateBrdf(float3(0.0, V.y, V.x), g_roughness)));
        
    g_output[dispatchThreadID.xy] = float4(col, 1);
}

float3 getCameraRay(float2 hCoords, float2 screenSize, float cosFovY, float sinFovY)
{
    float aspect = screenSize.y/screenSize.x;
    float sinFovX = sinFovY * aspect;

    float sa = sin(g_eyeAltitude);
    float ca = cos(g_eyeAltitude);
    float sz = sin(g_eyeAzimuth);
    float cz = cos(g_eyeAzimuth);

    float3 eye_z = float3(ca * sz, sa, -ca * cz);
    float3 eye_x = float3(cz, 0.0, sz);
    float3 eye_y = cross(eye_x, eye_z);

    return normalize(hCoords.x * sinFovY * eye_x + hCoords.y * sinFovX * eye_y + cosFovY * eye_z);
}

#define MATERIAL_TILES 0
#define MATERIAL_SHINY 1
#define MATERIAL_DIFFUSIVE 2
#define MATERIAL_EMISSIVE 3

Quad getLightSource()
{
    Quad quad;
    quad.c = float3(g_lightTransform[3][0], g_lightTransform[3][1], g_lightTransform[3][2]);
    quad.r = float3(g_lightTransform[0][0], g_lightTransform[0][1], g_lightTransform[0][2]);
    quad.u = float3(g_lightTransform[1][0], g_lightTransform[1][1], g_lightTransform[1][2]);
    quad.e = float2(g_lightWidth,g_lightHeight);
    return quad;
}

RayHit traceScene(Ray ray)
{
    Sphere sphere0;
    sphere0.o = float3(0,0,0);
    sphere0.r = 3;

    Sphere sphere1;
    sphere1.o = float3(10,-1.6,0);
    sphere1.r = 1;

    Sphere sphere2;
    sphere2.o = float3(6,-1.6,0);
    sphere2.r = 2;

    Plane plane;
    plane.n = normalize(float3(0,1,0));
    plane.d = 2.5;

    Quad quad = getLightSource();

    RayHit rh;
    rh.t = -1;
    rh.n = 0;

    RayHit th;

    #define Tr(x) { th = x; if (rh.t < 0.0 || (th.t >= 0.0 && th.t < rh.t)) rh = th; }
    Tr(sphereTrace(sphere0, ray, MATERIAL_SHINY));
    Tr(sphereTrace(sphere1, ray, MATERIAL_DIFFUSIVE));
    Tr(sphereTrace(sphere2, ray, MATERIAL_DIFFUSIVE));
    Tr(planeTrace(plane, ray, MATERIAL_TILES));
    Tr(quadTrace(quad, ray, MATERIAL_EMISSIVE));
    #undef Tr

    rh.n = rh.t > 0.0 && dot(rh.n, ray.d) < 0.0 ? rh.n : -rh.n;
    return rh;
}

float matTiles(float3 worldPos, float3 n)
{
    float2 r = (worldPos.xz + 100.0) * 0.2;
    int2 tileIds = (int2)r;
    return lerp(g_roughness, 0.7, (float)((tileIds.x ^ tileIds.y) & 0x1));
}

uint sampleRandomSeed(uint2 pixelCoord)
{
    float bn = g_blueNoise.Load(uint3((pixelCoord.xy) % 256, 0));
    return (uint)round(bn * 256.0 * 256.0);
}

float2 offsetFromSeed(uint seed)
{
    return float2((float)(seed % 256)/256.0, ((float)seed/256)/256.0);
}

void lighting(uint seed, float3 worldPos, float roughness, float3 n, float3 v, out float3 diff, out float3 spec)
{
    float2 sampleOffset = offsetFromSeed(seed);
    float3 radCol = 0.0;
    float3x3 basis = transpose(inventBasisFromNormal(n));
    diff = 0;
    spec = 0;

    uint rng = seed;

    uint specSamples = 0;
    uint diffSamples = 0;

    for (uint i = 0; i < g_lightSamples; ++i)
    {
        float2 uv = fmod(sampleHammersley(i , g_lightSamples) + sampleOffset, float2(1.0,1.0));
        float3 s = sampleCosineHemisphere(uv.x, uv.y);
        Ray r;
        r.d = mul(basis,s);
        r.o = worldPos + 0.0001 * r.d;

        float3 V = -v;
        float3 L = r.d;
        float3 H = normalize(L + V);
        float NdotV = max(dot(V, n), 0);
        float NdotH = max(dot(n, H), 0);
        float NdotL = max(dot(n, L), 0);
        float VdotH = max(dot(V, H), 0);
        float lambda = GetSmithJointGGXPartLambdaV(NdotV, roughness);
        float vis = DV_SmithJointGGX(NdotH, NdotL, max(NdotV,0), roughness, lambda);
        float weightOverPdf = 4.0 * vis * NdotL * VdotH / NdotH;
        float FweightOverPdf = weightOverPdf * F_Schlick(0.2, NdotV);
        bool isSpec = randomFloat(rng) < FweightOverPdf;
        RayHit rh = traceScene(r);
        if (isSpec)
            specSamples += 1;
        else
            diffSamples += 1;
        if (rh.t >= 0.0 && rh.materialID == MATERIAL_EMISSIVE)
        {
            if (isSpec)
                spec += g_lightIntensity * saturate(FweightOverPdf);
            else
                diff += g_lightIntensity;
        }
    
    }

    #if 1
    spec *= specSamples ? 1.0/(float)specSamples : 0;
    diff *= diffSamples ? 1.0/(float)diffSamples : 0;
    #else
    spec *= 1.0/g_lightSamples;
    diff *= 1.0/g_lightSamples;
    #endif
}

float3 linearToSRGB(float3 col)
{
    bool3 cutoff = col < 0.0031308.xxx;
    float3 higher = 1.055.xxx*pow(col, (1.0/2.4).xxx) - 0.055.xxx;
    float3 lower = col * 12.92.xxx;
    return lerp(higher, lower, cutoff);
}

[numthreads(8,8,1)]
void csRtScene(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (any(dispatchThreadID.xy > (int2)g_sizes.xy))
        return;

    uint seed = sampleRandomSeed(dispatchThreadID.xy);
    float2 hCoords = getHCoords(dispatchThreadID.xy);
    Ray ray;
    ray.o = g_eyePos;
    ray.d = getCameraRay(hCoords, g_sizes.xy, g_eyeCosFovY, g_eyeSinFovY);

    float roughness = 0.0;
    float3 col = 0;
    float3 alb = float3(0,0,0);
    RayHit rh = traceScene(ray);
    if (rh.t >= 0.0)
    {
        float3 e = 0;
        float3 worldPos = ray.o + ray.d * rh.t;
        switch (rh.materialID)
        {
        case MATERIAL_TILES:
            roughness = matTiles(worldPos, rh.n);
            alb = float3(0.8, 0.2, 0.1);
            break;
        case MATERIAL_SHINY:
            roughness = 0.7;
            alb = float3(0.1, 0.1, 0.6);
            break;
        case MATERIAL_DIFFUSIVE:
            roughness = 0.3;
            alb = float3(0.1, 0.3, 0.61);
            break;
        case MATERIAL_EMISSIVE:
            roughness = 0.0;
            e = g_lightIntensity;
            break;
        }

        float3 diff, spec;
        lighting(seed, worldPos, roughness, rh.n, ray.d, diff, spec);
        col = diff*alb + spec + e;
    }

    //g_output[dispatchThreadID.xy] = lerp(g_output[dispatchThreadID.xy], float4(linearToSRGB(col), 1), 0.04);
    g_output[dispatchThreadID.xy] = float4(linearToSRGB(col), 1.0);
}

