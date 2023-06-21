#include "brdf.hlsl"

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
    return normalize(float3(hCoords * float2(sinFovY, sinFovX), -cosFovY));
}

struct Ray
{
    float3 o;
    float3 d;
};

struct RayHit
{
    float t;
    float3 n;
};

struct Sphere
{
    float3 o;
    float r;
};

struct Plane
{
    float3 n;
    float d;
};

struct Quad
{
    float3 c; //center
    float3 r; //right vector (normalized)
    float3 u; //up vector (normalized)
    float2 e; //half extents (right & up)
};

RayHit sphereTrace(Sphere sphere, Ray ray)
{
    //x2 + y2 + z2 = r2
    //o + t*d = p
    //
    //(ox + t*dx)2 + (oy + t*dy)2 + (oz + t*dz)2 - r2 = 0
    //ox2 + 2toxdx + t2dx2 + oy2 + 2toydy + t2dy2 + oz2 + 2tozdz + t2dz2 - r2 = 0
    //t2*(dx2 + dy2 + dz2) + t*(2oxdx + 2oydy + 2ozdz) + ((ox2 + oy2 + oz2) - r2) = 0

    float3 o = ray.o - sphere.o;
    float a = dot(ray.d,ray.d);
    float b = 2.0 * dot(ray.d, o);
    float c = dot(o, o) - sphere.r*sphere.r;
    float b24ac = b*b - 4.0*a*c;

    RayHit rh;
    if (b24ac < 0.0)
    {
        rh.t = -1;
        rh.n = 0;
    }
    else
    {
        float inv2a = rcp(2.0*a);
        rh.t = (-b - sqrt(b24ac))*inv2a;
        rh.n = normalize((ray.o + ray.d * rh.t) - sphere.o);
    }

    return rh;
}

RayHit planeTrace(Plane p, Ray ray)
{
    // Plane trace line
    // ax + bx + cx = d
    // o + t*d = p
    //
    // (ox + t*dx)*a + (oy + t*dy)*b + (oz + t*dz)*c - d = 0
    // (aox + boy + coz - d) + t * (a*dx + b*dy + c*dz) = 0
    // t = (d - (aox + boy + coz)) / (adx + bdy + cdz) 

    float NDotO = dot(-p.n, ray.o);
    float NDotD = dot(-p.n, ray.d);

    RayHit rh;
    if (abs(NDotD) < 0.00001)
    {
        rh.t = -1;
        rh.n = 0;
    }
    else
    {
        rh.t = (p.d - NDotO) / NDotD;
        rh.n = p.n;
    }

    return rh;
}

RayHit quadTrace(Quad q, Ray ray)
{
    Plane p;
    p.n = cross(q.r,q.u);
    p.d = dot(-p.n, q.c);
    RayHit ph = planeTrace(p, ray);
    if (ph.t < 0.0)
        return ph;

    float3 hitPoint = (ph.t * ray.d + ray.o) - q.c;
    if (abs(dot(hitPoint, q.r)) > q.e.x || abs(dot(hitPoint, q.u)) > q.e.y)
    {
        ph.t = -1;
        return ph;
    }

    return ph;
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
    quad.c = float3(5,0,-4.2);
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

