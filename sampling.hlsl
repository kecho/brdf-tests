#ifndef __SAMPLING__H_
#define __SAMPLING__H_

#ifndef M_PI
#define M_PI 3.14159265359
#endif

// T. Duff, J. Burgess, P. Christensen, C. Hery, A. Kensler, M. Liani and R. Villemin
// Building an Orthonormal Basis, Revisited.
// Journal of Computer Graphics Techniques 6 (2017), 1, pp. 1-8.
float3x3 inventBasisFromNormal(float3 n)
{
	const float s  = (0.0f > n.z) ? -1.0f : 1.0f; //  std::copysignf(1.0f, n.z)
	const float a0 = -1.0f / (s + n.z);
	const float a1 = n.x * n.y * a0;

	const float3 t = { 1.0f + s * n.x * n.x * a0, s * a1, -s * n.x };
	const float3 b = { a1, s + n.y * n.y * a0, -n.y };

	return float3x3(t, b, n);
}

float radicalInverse(uint n, uint base)
{
	// Implementation from Physically Based Rendering.
	float val     = 0.0f;
	float invBase = 1.0f / base;
	float invBi   = invBase;

	while (n > 0)
	{
		uint d_i = (n % base);
		val += d_i * invBi;
		n *= invBase;
		invBi *= invBase;
	}

	return val;
}

float randomFloat(inout uint rng)
{
	rng = 214013*rng+2531011;
	return float(rng>>16)*(1.0f/65535.0f);
}

float2 sampleHalton(uint n)
{
	return float2(radicalInverse(n, 2), radicalInverse(n, 3));
}

uint randomUint(inout uint rng)
{
	rng=214013*rng+2531011;
	return (rng ^ rng>>16);
}

float2 sampleHammersley(uint i, uint n)
{
	// Van der Corput radical inverse: http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
	uint bits = i;
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	float vdc = float(bits) * 2.3283064365386963e-10f; // / 0x100000000
	return float2(float(i) / float(n), vdc);
}

float3 sampleUniformHemisphere(float u, float v)
{
	float phi = v * M_PI * 2.0f;
	float cosTheta = u;
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

	float sinPhi = sin(phi);
	float cosPhi = cos(phi);

	return float3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
}

float3 sampleCosineHemisphere(float u, float v)
{
	float phi = v * M_PI * 2.0f;
	float cosTheta = sqrt(u);
	float sinTheta = sqrt(1.0f - u);

	float sinPhi = sin(phi);
	float cosPhi = cos(phi);

	float x = cosPhi * sinTheta;
	float y = sinPhi * sinTheta;
	float z = cosTheta;

	return float3(x, y, z);
}

float3 sampleCosineHemisphere(float2 uv)
{
	return sampleCosineHemisphere(uv.x, uv.y);
}

#endif
