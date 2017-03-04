#define EPS (1e-10)

inline void body_body_interaction (
    float4 bi,
    float4 bj,
    float4 * ai
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
    global float4 * global_p,
    global float4 * global_a,
    global const int * points
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
