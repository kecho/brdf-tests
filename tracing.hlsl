#ifndef __TRACING__H__
#define __TRACING__H__

struct Ray
{
    float3 o;
    float3 d;
};

struct RayHit
{
    float t;
    float3 n;
    uint materialID;
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

RayHit sphereTrace(Sphere sphere, Ray ray, uint materialID = -1)
{
    //x2 + y2 + z2 = r2
    //o + t*d = p
    //
    //(ox + t*dx)2 + (oy + t*dy)2 + (oz + t*dz)2 - r2 = 0
    //ox2 + 2toxdx + t2dx2 + oy2 + 2toydy + t2dy2 + oz2 + 2tozdz + t2dz2 - r2 = 0
    //t2*(dx2 + dy2 + dz2) + t*(2oxdx + 2oydy + 2ozdz) + ((ox2 + oy2 + oz2) - r2) = 0

    ray.d = normalize(ray.d);

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
        rh.materialID = -1;
    }
    else
    {
        float inv2a = rcp(2.0*a);
        rh.t = (-b - sqrt(b24ac))*inv2a;
        rh.n = normalize((ray.o + ray.d * rh.t) - sphere.o);
        rh.materialID = materialID;
    }

    return rh;
}

RayHit planeTrace(Plane p, Ray ray, uint materialID = -1)
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
        rh.materialID = -1;
    }
    else
    {
        rh.t = (p.d - NDotO) / NDotD;
        rh.n = p.n;
        rh.materialID = materialID;
    }

    return rh;
}

RayHit quadTrace(Quad q, Ray ray, uint materialID = -1)
{
    Plane p;
    p.n = cross(q.r,q.u);
    p.d = dot(-p.n, q.c);
    RayHit ph = planeTrace(p, ray);
    ph.materialID = materialID;
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

#endif
