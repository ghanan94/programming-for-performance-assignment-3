#define EPS (1e-10)
#define BIN_LENGTH (100.0f)
#define BINS_PER_DIM (10)

#define IS_IN(min_value, max_value, value)  \
    (((value) >= (min_value)) && ((value) < (max_value)))

typedef float4 (bins_t)[BINS_PER_DIM][BINS_PER_DIM];
typedef int (bin_pts_offsets_t)[BINS_PER_DIM][BINS_PER_DIM];

inline void get_global_ids (
    int * const global_id
    )
{
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
}


//
// Get sort out the points for each bin in a 1-D Array.
// Doing this buy simply placing points in order of
// what bin they are in. AN offset for each bin is used
// to get the first index in the 1D array for the first
// point
//
__kernel void construct_bin_pts (
    global int * const global_bin_pts,
    global bin_pts_offsets_t * const global_bin_pts_offsets,
    global float4 const * const global_p,
    global int const * const points,
    global bins_t const * const global_cm
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
    int counter;
    int offset;
    int idx;
    global float4 const * temp_pt;

    get_global_ids(global_id);

    idx = global_id[0] * (BINS_PER_DIM * BINS_PER_DIM) + global_id[1] * BINS_PER_DIM + global_id[2];


    //
    // Calculate offset from beginning of bin_pts array to the beginning
    // of where this bin's points start
    //
    offset = 0;
    temp_pt = (global int *) global_cm;

    for (i = 0; i < idx; ++i)
    {
        offset += (int) temp_pt[i].w;
    }

    global_bin_pts_offsets[global_id[0]][global_id[1]][global_id[2]] = offset;

    //
    // Calculate bounds for the bin
    //
    min_x = (float) (global_id[0] * BIN_LENGTH);
    max_x = min_x + BIN_LENGTH;

    min_y = (float) (global_id[1] * BIN_LENGTH);
    max_y = min_y + BIN_LENGTH;

    min_z = (float) (global_id[2] * BIN_LENGTH);
    max_z = min_z + BIN_LENGTH;

    //
    // Iterate through all the points and find the points that should lie within this bin
    //
    counter = 0;

    for (i = 0; i < points[0]; ++i)
    {
        if (IS_IN(min_x, max_x, global_p[i].x)
            && IS_IN(min_y, max_y, global_p[i].y)
            && IS_IN(min_z, max_z, global_p[i].z))
        {
            global_bin_pts[offset + counter] = i;
            counter++;
        }
    }
}

//
// Calculate center mass for a bin
//
__kernel void calculate_bins_cm (
    global bins_t * const global_cm,
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
    float4 val;

    get_global_ids(global_id);

    //
    // Calculate bounds for the bin
    //
    min_x = (float) (global_id[0] * BIN_LENGTH);
    max_x = min_x + BIN_LENGTH;

    min_y = (float) (global_id[1] * BIN_LENGTH);
    max_y = min_y + BIN_LENGTH;

    min_z = (float) (global_id[2] * BIN_LENGTH);
    max_z = min_z + BIN_LENGTH;

    val = (float4) {0.0f, 0.0f, 0.0f, 0.0f};

    //
    // Iterate through all the points and find the points that should lie within this bin
    //
    for (i = 0; i < points[0]; ++i)
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

    global_cm[global_id[0]][global_id[1]][global_id[2]] = val;
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
    global bins_t const * const global_cm,
    global int const * const global_bin_pts,
    global bin_pts_offsets_t const * const global_bin_pts_offsets,
    global float4 * const global_a,
    global int const * const points
    )
{
    int global_id;
    int i;
    int x;
    int y;
    int z;
    float4 my_position;
    float4 acc;
    int offset;
    float4 neg_bin;
    int x_bin;
    int y_bin;
    int z_bin;

    global_id = get_global_id(0);
    my_position = global_p[global_id];

    x_bin = ((int) my_position.x) / ((int) BIN_LENGTH);
    y_bin = ((int) my_position.y) / ((int) BIN_LENGTH);
    z_bin = ((int) my_position.z) / ((int) BIN_LENGTH);

    acc = (float4) {0.0f, 0.0f, 0.0f, 0.0f};

    //
    // Bin approx for all bins
    //
    for (x = 0; x < BINS_PER_DIM; ++x)
    {
        for (y = 0; y < BINS_PER_DIM; ++y)
        {
            for (z = 0; z < BINS_PER_DIM; ++z)
            {
                body_body_interaction(my_position, global_cm[x][y][z], &acc);
            }
        }
    }

    //
    // Subtract near bins
    //
    if (x_bin > 0)
    {
        x = x_bin - 1;
    }
    else
    {
        x = 0;
    }

    for (; x < x_bin + 2 && x < BINS_PER_DIM; ++x)
    {

        if (y_bin > 0)
        {
            y = y_bin - 1;
        }
        else
        {
            y = 0;
        }

        for (; y < y_bin + 2 && y < BINS_PER_DIM; ++y)
        {

            if (z_bin > 0)
            {
                z = z_bin - 1;
            }
            else
            {
                z = 0;
            }

            for (; z < z_bin + 2 && z < BINS_PER_DIM; ++z)
            {
                neg_bin.x = 2 * my_position.x - global_cm[x][y][z].x;
                neg_bin.y = 2 * my_position.y - global_cm[x][y][z].y;
                neg_bin.z = 2 * my_position.z - global_cm[x][y][z].z;
                neg_bin.w = global_cm[x][y][z].w;

                body_body_interaction(my_position, neg_bin, &acc);


                offset = global_bin_pts_offsets[x][y][z];

                for (i = 0; i < ((int) global_cm[x][y][z].w); ++i)
                {
                    body_body_interaction(my_position, global_p[global_bin_pts[offset] + i], &acc);
                }
            }
        }
    }

    global_a[global_id] = acc;
}
