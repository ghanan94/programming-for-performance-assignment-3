/* nbody simulation, version 0 */
/* Modified by Patrick Lam; original source: GPU Gems, Chapter 31 */

#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define EPS 1e-10
#define BIN_LENGTH (100.0f)
#define BINS_PER_DIM (10)

/// runtime on my 2011 computer: 1m; in 2013, 27s.
// on my 2011 laptop, 1m34s
#define POINTS 5000 * 64
#define SPACE 1000.0f;

#define IS_IN(min_value, max_value, value)  \
    (((value) >= (min_value)) && ((value) < (max_value)))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

cl_float4 cm[BINS_PER_DIM][BINS_PER_DIM][BINS_PER_DIM];
int bin_pts_offsets[BINS_PER_DIM][BINS_PER_DIM][BINS_PER_DIM];
cl_float4 bin_pts[POINTS];

void construct_bins_cm (
    cl_float4 const * const global_p,
    int const points,
    cl_float4 (* const global_cm)[BINS_PER_DIM][BINS_PER_DIM]
    )
{
    for (int x = 0; x < BINS_PER_DIM; ++x)
    {
        for (int y = 0; y < BINS_PER_DIM; ++y)
        {
            for (int z = 0; z < BINS_PER_DIM; ++z)
            {
                float min_x, min_y, min_z;
                float max_x, max_y, max_z;
                cl_float4 val;

                //
                // Calculate bounds for the bin
                //
                min_x = (float) (x * BIN_LENGTH);
                max_x = min_x + BIN_LENGTH;

                min_y = (float) (y * BIN_LENGTH);
                max_y = min_y + BIN_LENGTH;

                min_z = (float) (z * BIN_LENGTH);
                max_z = min_z + BIN_LENGTH;

                val = (cl_float4) {0.0f, 0.0f, 0.0f, 0.0f};

                //
                // Iterate through all the points and find the points that should lie within this bin
                //
                for (int i = 0; i < points; ++i)
                {
                    if (IS_IN(min_x, max_x, global_p[i].x)
                        && IS_IN(min_y, max_y, global_p[i].y)
                        && IS_IN(min_z, max_z, global_p[i].z))
                    {
                        val.x += global_p[i].x;
                        val.y += global_p[i].y;
                        val.z += global_p[i].z;
                        val.w += 1.0f;
                    }
                }

                val.x /= val.w;
                val.y /= val.w;
                val.z /= val.w;

                global_cm[x][y][z] = val;
            }
        }
    }
}

void construct_bin_pts (
    cl_float4 * const global_bin_pts,
    int (* const global_bin_pts_offsets)[BINS_PER_DIM][BINS_PER_DIM],
    cl_float4 const * const global_p,
    int const points,
    cl_float4 (* const global_cm)[BINS_PER_DIM][BINS_PER_DIM]
    )
{
    cl_float4 * global_cm_linearized;

    global_cm_linearized = (cl_float4 *) global_cm;

    for (int x = 0; x < BINS_PER_DIM; ++x)
    {
        for (int y = 0; y < BINS_PER_DIM; ++y)
        {
            for (int z = 0; z < BINS_PER_DIM; ++z)
            {
                int offset;
                float min_x, min_y, min_z;
                float max_x, max_y, max_z;
                int counter;
                int idx;

                idx = (x * BINS_PER_DIM * BINS_PER_DIM) +
                    (y * BINS_PER_DIM) + z;

                //
                // Calculate offset from beginning of bin_pts array to the beginning
                // of where this bin's points start
                //
                offset = 0;
                for (int i = 0; i < idx; ++i)
                {
                    offset += (int) global_cm_linearized[i].w;
                }

                global_bin_pts_offsets[x][y][z] = offset;

                //
                // Calculate bounds for the bin
                //
                min_x = (float) (x * BIN_LENGTH);
                max_x = min_x + BIN_LENGTH;

                min_y = (float) (y * BIN_LENGTH);
                max_y = min_y + BIN_LENGTH;

                min_z = (float) (z * BIN_LENGTH);
                max_z = min_z + BIN_LENGTH;

                counter = 0;

                //
                // Iterate through all the points and find the points that should lie within this bin
                //
                for (int i = 0; i < points; ++i)
                {
                    if (IS_IN(min_x, max_x, global_p[i].x)
                        && IS_IN(min_y, max_y, global_p[i].y)
                        && IS_IN(min_z, max_z, global_p[i].z))
                    {
                        global_bin_pts[offset + counter] = global_p[i];
                        counter++;
                    }
                }
            }
        }
    }
}

void body_body_interaction(cl_float4 bi, cl_float4 bj, cl_float4 *ai) {
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

void calculateForces (
    int points,
    int global_id,
    cl_float4 * global_p,
    cl_float4 * global_a,
    cl_float4 const (* const global_cm)[BINS_PER_DIM][BINS_PER_DIM],
    cl_float4 const * const global_bin_pts,
    int const (* const global_bin_pts_offsets)[BINS_PER_DIM][BINS_PER_DIM]
    )
{
    cl_float4 my_position = global_p[global_id];
    cl_float4 acc = {{0.0f, 0.0f, 0.0f, 1.0f}};
    int x_bin, y_bin, z_bin;
    cl_float4 const * global_cm_linear;

    global_cm_linear = (cl_float4 *) global_cm;

    x_bin = (int) (my_position.x / BIN_LENGTH);
    y_bin = (int) (my_position.y / BIN_LENGTH);
    z_bin = (int) (my_position.z / BIN_LENGTH);

    //
    // Bin approx for all bins
    //
    for (int i = 0; i < (BINS_PER_DIM * BINS_PER_DIM * BINS_PER_DIM); ++i)
    {
        body_body_interaction(my_position, global_cm_linear[i], &acc);
    }

    for (int x = MAX(0, x_bin - 1); x < MIN(BINS_PER_DIM, x_bin + 2); ++x)
    {
        for (int y = MAX(0, y_bin - 1); y < MIN(BINS_PER_DIM, y_bin + 2); ++y)
        {
            for (int z = MAX(0, z_bin - 1); z < MIN(BINS_PER_DIM, z_bin + 2); ++z)
            {
                cl_float4 neg_bin;
                int offset;

                neg_bin.x = 2 * my_position.x - global_cm[x][y][z].x;
                neg_bin.y = 2 * my_position.y - global_cm[x][y][z].y;
                neg_bin.z = 2 * my_position.z - global_cm[x][y][z].z;
                neg_bin.w = global_cm[x][y][z].w;

                body_body_interaction(my_position, neg_bin, &acc);

                offset = global_bin_pts_offsets[x][y][z];

                for (int i = 0; i < ((int) global_cm[x][y][z].w); ++i)
                {
                    body_body_interaction(my_position, global_bin_pts[offset + i], &acc);
                }
            }
        }
    }

    global_a[global_id] = acc;
}

cl_float4 * initializePositions() {
    cl_float4 * pts = (cl_float4*) malloc(sizeof(cl_float4)*POINTS);
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
    cl_float4 * pts = (cl_float4*) malloc(sizeof(cl_float4)*POINTS);
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

    construct_bins_cm(x, POINTS, (cl_float4 (*)[BINS_PER_DIM][BINS_PER_DIM]) &cm);
    construct_bin_pts((cl_float4 *) &bin_pts,
                      (int (*)[BINS_PER_DIM][BINS_PER_DIM]) &bin_pts_offsets,
                      x,
                      POINTS,
                      (cl_float4 (*)[BINS_PER_DIM][BINS_PER_DIM]) &cm);

    for (int i = 0; i < POINTS; i++)
    {
        calculateForces(POINTS, i, x, a,
                        (cl_float4 (*)[BINS_PER_DIM][BINS_PER_DIM]) &cm,
                        (cl_float4 *) &bin_pts,
                        (int (*)[BINS_PER_DIM][BINS_PER_DIM]) &bin_pts_offsets);
    }

    for (int i = 0; i < POINTS; i++)
    printf("(%2.2f,%2.2f,%2.2f,%2.2f) (%2.3f,%2.3f,%2.3f)\n",
           x[i].x, x[i].y, x[i].z, x[i].w,
           a[i].x, a[i].y, a[i].z);

    free(x);
    free(a);
    return 0;
}
