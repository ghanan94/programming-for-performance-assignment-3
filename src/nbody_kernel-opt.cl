#define EPS (1e-10)
#define BIN_LENGTH (100.0f)
#define BINS_PER_DIM (10)

#define IS_IN(min_value, max_value, value)  \
    (((value) >= (min_value)) && ((value) <= (max_value)))

typedef float4 (bins_t)[BINS_PER_DIM][BINS_PER_DIM];


__kernel void calculate_bins_cm (
    global bins_t * const cm,
    global float4 const * const global_p,
    global int const * const points
    )
{
    int global_id[3];
    int i;
    float max_x;
    float min_x;
    float max_y;
    float min_y;
    float max_z;
    float min_z;

    //
    // x dim
    //
    global_id[0] = get_global_id(0);

    //
    // y dim
    //
    global_id[1] = get_global_id(1);

    //
    // z dim
    //
    global_id[2] = get_global_id(2);

    //
    // Calculate bounds for the bin
    //
    min_x = (float) (global_id[0] * BIN_LENGTH);
    max_x = min_x + BIN_LENGTH;

    min_y = (float) (global_id[1] * BIN_LENGTH);
    max_y = min_x + BIN_LENGTH;

    min_z = (float) (global_id[2] * BIN_LENGTH);
    max_z = min_x + BIN_LENGTH;

    //
    // Iterate through all the points and find the points that should lie within this bin
    //
    for (i = 0; i < points[0]; ++i)
    {
        if (IS_IN(min_x, max_x, global_p[i].x)
            && IS_IN(min_y, max_y, global_p[i].y)
            && IS_IN(min_z, max_z, global_p[i].z))
        {
            cm[global_id[0]][global_id[1]][global_id[2]].x += global_p[i].x;
            cm[global_id[0]][global_id[1]][global_id[2]].y += global_p[i].y;
            cm[global_id[0]][global_id[1]][global_id[2]].z += global_p[i].z;
            cm[global_id[0]][global_id[1]][global_id[2]].w += 1.0f;
        }
    }

    cm[global_id[0]][global_id[1]][global_id[2]].x /= cm[global_id[0]][global_id[1]][global_id[2]].w;
    cm[global_id[0]][global_id[1]][global_id[2]].y /= cm[global_id[0]][global_id[1]][global_id[2]].w;
    cm[global_id[0]][global_id[1]][global_id[2]].z /= cm[global_id[0]][global_id[1]][global_id[2]].w;
}

inline void body_body_interaction (
    float4 const bi,
    float4 const bj,
    float4 * const ai
    )
{
    float4 r;
    float dist_sqr;
    float dist_sixth;
    float inv_dist_cube;
    float s;


    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    r.w = 1.0f;

    dist_sqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS;

    dist_sixth = dist_sqr * dist_sqr * dist_sqr;
    inv_dist_cube = 1.0f / sqrt(dist_sixth);

    s = bj.w * inv_dist_cube;

    ai->x += r.x * s;
    ai->y += r.y * s;
    ai->z += r.z * s;
}

__kernel void nbody (
    global float4 const * const global_p,
    global float4 * const global_a,
    global int const * const points
    )
{
    int global_id;
    int i;
    float4 my_position;
    float4 acc;

    global_id = get_global_id(0);
    my_position = global_p[global_id];

    for (i = 0; i < points[0]; ++i)
    {
        body_body_interaction(my_position, global_p[i], &acc);
    }

    global_a[global_id] = acc;
}
