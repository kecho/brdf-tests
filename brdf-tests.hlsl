#include "brdf.hlsl"
#include "tracing.hlsl"

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
}

#define g_VdotN g_angles.x
#define g_roughness g_material.x
#define g_eyePos g_eyeArgs0.xyz
#define g_eyeAzimuth g_eyeArgs0.w
#define g_eyeAltitude g_eyeArgs1.x
#define g_eyeCosFovY g_eyeArgs1.y
#define g_eyeSinFovY g_eyeArgs1.z

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

float3 drawBrdf(float2 o, float lineThickness, float2 V, float2 N, float roughness, float2 coord, out float intValue)
{
    uint i = 0;
    float3 col = float3(0,0,0);
    intValue = 0.0;

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
        intValue += ww;

        col += drawLine(o, o + ww * L, lineThickness, float3(0.0, 0.0, 0.4), coord);
    }

    intValue /= SAMPLE_LINES;

    return col;
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
    col += drawBrdf(o, lineThickness, float2(V.x, V.y), N, g_roughness, coord, brdfV);

    if (dispatchThreadID.x == 0 && dispatchThreadID.y == 0)
        g_brdfBuff.Store(0, asuint(brdfV));
        
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
    //return normalize(float3(, hCoords.y * sinFovX, -cosFovY));
}

RayHit traceScene(Ray ray)
{
    Sphere sphere;
    sphere.o = float3(0,0,-8);
    sphere.r = 4;

    Plane plane;
    plane.n = normalize(float3(0,1,0));
    plane.d = 2.5;

    Quad quad;
    quad.c = float3(0,0,-4.2);
    quad.r = float3(1,0,0);
    quad.u = float3(0,1,0);
    quad.e = float2(2,2);

    RayHit rh;
    rh.t = -1;
    rh.n = 0;

    RayHit th;

    th = sphereTrace(sphere, ray);
    if (rh.t < 0.0 || (th.t >= 0.0 && th.t < rh.t))
        rh = th;

    th = planeTrace(plane, ray);
    if (rh.t < 0.0 || (th.t >= 0.0 && th.t < rh.t))
        rh = th;

    th = quadTrace(quad, ray);
    if (rh.t < 0.0 || (th.t >= 0.0 && th.t < rh.t))
        rh = th;

    return rh;
}

[numthreads(8,8,1)]
void csRtScene(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float2 hCoords = getHCoords(dispatchThreadID.xy);
    Ray ray;
    ray.o = g_eyePos;
    ray.d = getCameraRay(hCoords, g_sizes.xy, g_eyeCosFovY, g_eyeSinFovY);

    float3 col = float3(0,0,0);
    RayHit rh = traceScene(ray);
    if (rh.t >= 0.0)
        col = max(0, dot(-ray.d, rh.n));

    g_output[dispatchThreadID.xy] = float4(col, 1);
}

