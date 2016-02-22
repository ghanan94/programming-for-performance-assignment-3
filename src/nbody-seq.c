/* nbody simulation, version 0 */
/* Modified by Patrick Lam; original source: GPU Gems, Chapter 31 */

#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define EPS 1e-10

/// runtime on my 2011 computer: 1m; in 2013, 27s.
// on my 2011 laptop, 1m34s
#define POINTS 500 * 64
#define SPACE 1000.0f;

void bodyBodyInteraction(cl_float4 bi, cl_float4 bj, cl_float4 *ai) {
    cl_float4 r;
    
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    r.w = 1.0f;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS;

    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);

    float s = bj.w * invDistCube;

    ai->x += r.x * s;
    ai->y += r.y * s;
    ai->z += r.z * s;
}

void calculateForces(int points, int global_id, cl_float4 * globalP, cl_float4 * globalA) {
    cl_float4 myPosition = globalP[global_id];
    int i;

    cl_float4 acc = {{0.0f, 0.0f, 0.0f, 1.0f}};
    
    for (i = 0; i < points; i ++) {
	bodyBodyInteraction(myPosition, globalP[i], &acc);
    }
    globalA[global_id] = acc;
}

cl_float4 * initializePositions() {
    cl_float4 * pts = malloc(sizeof(cl_float4)*POINTS);
    int i;

    srand(42L); // for deterministic results

    for (i = 0; i < POINTS; i++) {
	// quick and dirty generation of points
	// not random at all, but I don't care.
	pts[i].x = ((float)rand())/RAND_MAX * SPACE;
	pts[i].y = ((float)rand())/RAND_MAX * SPACE;
	pts[i].z = ((float)rand())/RAND_MAX * SPACE;
	pts[i].w = 1.0f; // size = 1.0f for simplicity.
    }
    return pts;
}

cl_float4 * initializeAccelerations() {
    cl_float4 * pts = malloc(sizeof(cl_float4)*POINTS);
    int i;

    for (i = 0; i < POINTS; i++) {
	pts[i].x = pts[i].y = pts[i].z = pts[i].w = 0;
    }
    return pts;
}

int main(int argc, char ** argv)
{
    cl_float4 * x = initializePositions();
    cl_float4 * a = initializeAccelerations();

    int i;
    for (i = 0; i < POINTS; i++)
	calculateForces(POINTS, i, x, a);

    for (i = 0; i < POINTS; i++)
	printf("(%2.2f,%2.2f,%2.2f,%2.2f) (%2.3f,%2.3f,%2.3f)\n", 
	       x[i].x, x[i].y, x[i].z, x[i].w,
	       a[i].x, a[i].y, a[i].z);
    free(x);
    free(a);
    return 0;
}
