RWTexture2D<float4> g_output : register(u0);
RWByteAddressBuffer g_brdfBuff : register(u1);

#ifndef PI
#define PI 3.14159265359
#endif

#ifndef INV_PI
#define INV_PI (1.0/PI)
#endif

#ifndef SAMPLE_LINES
#define SAMPLE_LINES 64
#endif

float F_Schlick(float f0, float f90, float u)
{
    float x = 1.0 - u;
    float x2 = x * x;
    float x5 = x * x2 * x2;
    return (f90 - f0) * x5 + f0;                // sub mul mul mul sub mad
}

float F_Schlick(float f0, float u)
{
    return F_Schlick(f0, 1.0, u);               // sub mul mul mul sub mad
}

float GetSmithJointGGXPartLambdaV(float NdotV, float roughness)
{
    float a2 = roughness * roughness;
    return sqrt((-NdotV * a2 + NdotV) * NdotV + a2);
}

// Inline D_GGX() * V_SmithJointGGX() together for better code generation.
float DV_SmithJointGGX(float NdotH, float NdotL, float NdotV, float roughness, float partLambdaV)
{
    float a2 = roughness*roughness;
    float s = (NdotH * a2 - NdotH) * NdotH + 1.0;

    float lambdaV = NdotL * partLambdaV;
    float lambdaL = NdotV * sqrt((-NdotL * a2 + NdotL) * NdotL + a2);

    float2 D = float2(a2, s * s);            // Fraction without the multiplier (1/Pi)
    float2 G = float2(1, lambdaV + lambdaL); // Fraction without the multiplier (1/2)

    // This function is only used for direct lighting.
    // If roughness is 0, the probability of hitting a punctual or directional light is also 0.
    // Therefore, we return 0. The most efficient way to do it is with a max().
    return INV_PI * 0.5 * (D.x * G.x) / max(D.y * G.y, 0.00001);
}

cbuffer constants : register(b0)
{
    float4 g_sizes;
    float4 g_angles;
    float4 g_material;
    float4 g_coordTransform;
}

#define g_VdotN g_angles.x
#define g_roughness g_material.x

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

float3 drawLine2(float2 p0, float2 p1, float thickness, float3 col, float2 coord, float3 bCol)
{
    return bCol;
}

float2 hemisphereSample(uint i, uint count)
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
        float2 L = hemisphereSample(i, SAMPLE_LINES);
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

float2 getCoord(uint2 pixelCoord)
{
    float aspect = g_sizes.x/g_sizes.y;
    float2 uv = (pixelCoord + 0.5) * g_sizes.zw;
    float2 coord = uv * 2.0 - 1.0;
    coord.y *= -1;
    coord.x *= aspect;

    coord.xy += g_coordTransform.xy;
    coord.xy /= g_coordTransform.z;
    return coord;
}

[numthreads(8,8,1)]
void csBrdfTestsMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float2 coord = getCoord(dispatchThreadID.xy);

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
        
    g_output[dispatchThreadID.xy] = float4(col,1);
}
