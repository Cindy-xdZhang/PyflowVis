__device__ inline double2 make_d2(double x, double y){ double2 a; a.x=x; a.y=y; return a; }
__device__ inline double2 d2_add(const double2& a, const double2& b){ return make_d2(a.x+b.x, a.y+b.y); }
__device__ inline double2 d2_sub(const double2& a, const double2& b){ return make_d2(a.x-b.x, a.y-b.y); }
__device__ inline double2 d2_smul(double s, const double2& a){ return make_d2(s*a.x, s*a.y); }
__device__ inline double  d2_norm(const double2& a){ return sqrt(a.x*a.x + a.y*a.y); }
__device__ inline double cuda_max(double a, double b){ return (a > b) ? a : b; }
__device__ inline double cuda_min(double a, double b){ return (a < b) ? a : b; }
__device__ inline int pos_to_prev_idx_vertex_device(double pos, double dx){ return (int)floor(pos / dx); }
__device__ inline bool is_inbounds_2D_vertex_device(int width, int height, double dx, double dy, double2 p){
    return (p.x > 0.0) && (p.x < (width - 1)*dx) && (p.y > 0.0) && (p.y < (height - 1)*dy);
}

// Metadata for the preloaded sub-tile
struct TileDesc {
    int ox; // start x index
    int oy; // start y index
    int ot; // start t index
    int sx; // tile width  (#samples in x)
    int sy; // tile height (#samples in y)
    int st; // tile time length (#timesteps)
};

__device__ inline bool tile_has_data(const TileDesc& tile){
    return tile.sx > 0 && tile.sy > 0 && tile.st > 0;
}

__device__ inline bool tile_contains(const TileDesc& tile, int ix, int iy, int it){
    return (ix >= tile.ox && ix < tile.ox + tile.sx) &&
           (iy >= tile.oy && iy < tile.oy + tile.sy) &&
           (it >= tile.ot && it < tile.ot + tile.st);
}

// Linear index inside the shared-memory tile
__device__ inline int tile_linear_idx(const TileDesc& tile, int ix, int iy, int it){
    int lx = ix - tile.ox;
    int ly = iy - tile.oy;
    int lt = it - tile.ot;
    return lt*tile.sy*tile.sx + ly*tile.sx + lx;
}

__device__ double2 SpatialInterpolate2DUnsteadyField_device(float* u, float* v, double2 pos, int width, int height, int TotalTimeSteps, double dx, double dy, int it){
    int ix = pos_to_prev_idx_vertex_device(pos.x, dx);
    int iy = pos_to_prev_idx_vertex_device(pos.y, dy);
    if ((ix < 0) || (ix > width - 1) || (iy < 0) || (iy > height - 1) || (it < 0) || (it > TotalTimeSteps - 1))
        return make_d2(0.0, 0.0);
    int tl = it*height*width + iy*width + ix;
    bool xlimit = !(ix < width - 1);
    bool ylimit = !(iy < height - 1);
    int tr = xlimit ? tl : tl + 1;
    int bl = ylimit ? tl : tl + width;
    int br = ylimit ? tr : tr + width;
    double2 v_tl = make_d2(u[tl], v[tl]);
    double2 v_tr = make_d2(u[tr], v[tr]);
    double2 v_bl = make_d2(u[bl], v[bl]);
    double2 v_br = make_d2(u[br], v[br]);
    double x_alpha = (pos.x - ix*dx) / dx;
    double y_alpha = (pos.y - iy*dy) / dy;
    double2 top = d2_add(d2_smul(1.0 - x_alpha, v_tl), d2_smul(x_alpha, v_tr));
    double2 bot = d2_add(d2_smul(1.0 - x_alpha, v_bl), d2_smul(x_alpha, v_br));
    return d2_add(d2_smul(1.0 - y_alpha, top), d2_smul(y_alpha, bot));
}

// Interpolate using the shared-memory tile when available; otherwise fall back to global
__device__ double2 SpatialInterpolate2DUnsteadyField_device_tiled(float* u, float* v, const TileDesc& tile,
    float* tile_u, float* tile_v, double2 pos, int width, int height, int TotalTimeSteps, double dx, double dy, int it){

    int ix = pos_to_prev_idx_vertex_device(pos.x, dx);
    int iy = pos_to_prev_idx_vertex_device(pos.y, dy);

    // Try the shared-memory tile first
    if (tile_has_data(tile) && tile_contains(tile, ix, iy, it)){
        // Neighbor vertex indices with boundary handling
        int ix1 = cuda_min(ix + 1, tile.ox + tile.sx - 1);
        int iy1 = cuda_min(iy + 1, tile.oy + tile.sy - 1);

        int tl_idx = tile_linear_idx(tile, ix,  iy,  it);
        int tr_idx = tile_linear_idx(tile, ix1, iy,  it);
        int bl_idx = tile_linear_idx(tile, ix,  iy1, it);
        int br_idx = tile_linear_idx(tile, ix1, iy1, it);

        double2 v_tl = make_d2(tile_u[tl_idx], tile_v[tl_idx]);
        double2 v_tr = make_d2(tile_u[tr_idx], tile_v[tr_idx]);
        double2 v_bl = make_d2(tile_u[bl_idx], tile_v[bl_idx]);
        double2 v_br = make_d2(tile_u[br_idx], tile_v[br_idx]);

        double x_alpha = (pos.x - ix*dx) / dx;
        double y_alpha = (pos.y - iy*dy) / dy;
        double2 top = d2_add(d2_smul(1.0 - x_alpha, v_tl), d2_smul(x_alpha, v_tr));
        double2 bot = d2_add(d2_smul(1.0 - x_alpha, v_bl), d2_smul(x_alpha, v_br));
        return d2_add(d2_smul(1.0 - y_alpha, top), d2_smul(y_alpha, bot));
    }

    // Fallback to global memory
    return SpatialInterpolate2DUnsteadyField_device(u, v, pos, width, height, TotalTimeSteps, dx, dy, it);
}

__device__ double2 Interpolate2DUnsteadyField_device(float* u, float* v, double2 pos, int width, int height, int TotalTimeSteps, double dx, double dy, double dt, double t){
    if (t < 0.0){ if (fabs(t) < 1e-9) t = 0.0; }
    int t_1 = pos_to_prev_idx_vertex_device(t, dt);
    if (t_1 >= (TotalTimeSteps - 1))
        return SpatialInterpolate2DUnsteadyField_device(u, v, pos, width, height, TotalTimeSteps, dx, dy, TotalTimeSteps - 1);
    double2 v_1 = SpatialInterpolate2DUnsteadyField_device(u, v, pos, width, height, TotalTimeSteps, dx, dy, t_1);
    double2 v_2 = SpatialInterpolate2DUnsteadyField_device(u, v, pos, width, height, TotalTimeSteps, dx, dy, t_1 + 1);
    double t_alpha = (t - t_1*dt) / dt;
    return d2_add(d2_smul(1.0 - t_alpha, v_1), d2_smul(t_alpha, v_2));
}

__device__ double2 Interpolate2DUnsteadyField_device_tiled(float* u, float* v, const TileDesc& tile,
    float* tile_u, float* tile_v, double2 pos, int width, int height, int TotalTimeSteps, double dx, double dy, double dt, double t){
    if (t < 0.0){ if (fabs(t) < 1e-9) t = 0.0; }
    int t_1 = pos_to_prev_idx_vertex_device(t, dt);
    if (t_1 >= (TotalTimeSteps - 1))
        return SpatialInterpolate2DUnsteadyField_device_tiled(u, v, tile, tile_u, tile_v, pos, width, height, TotalTimeSteps, dx, dy, TotalTimeSteps - 1);

    double2 v_1 = SpatialInterpolate2DUnsteadyField_device_tiled(u, v, tile, tile_u, tile_v, pos, width, height, TotalTimeSteps, dx, dy, t_1);
    double2 v_2 = SpatialInterpolate2DUnsteadyField_device_tiled(u, v, tile, tile_u, tile_v, pos, width, height, TotalTimeSteps, dx, dy, t_1 + 1);
    double t_alpha = (t - t_1*dt) / dt;
    return d2_add(d2_smul(1.0 - t_alpha, v_1), d2_smul(t_alpha, v_2));
}

__device__ double2 advect_pathline_2D_rk4_device(float* u, float* v, int w, int h, int TotalTimeSteps, double dx, double dy, double dt,
    double x0, double y0, double t_i, double FTLE_dt, int FTLE_steps){
    double2 p = make_d2(x0, y0);

    //rk4 integration
    for (int k = 0; k < FTLE_steps; ++k){
        double current_time = t_i + k*FTLE_dt;
        if (current_time < 1e-15 && FTLE_dt < 0) return p;

        double2 a = Interpolate2DUnsteadyField_device(u, v, p, w, h, TotalTimeSteps, dx, dy, dt, current_time);
        double2 step1 = d2_add(p, d2_smul(0.5*FTLE_dt, a));
        double2 b = Interpolate2DUnsteadyField_device(u, v, step1, w, h, TotalTimeSteps, dx, dy, dt, current_time + FTLE_dt*0.5);
        double2 step2 = d2_add(p, d2_smul(0.5*FTLE_dt, b));
        double2 c = Interpolate2DUnsteadyField_device(u, v, step2, w, h, TotalTimeSteps, dx, dy, dt, current_time + FTLE_dt*0.5);
        double2 step3 = d2_add(p, d2_smul(0.5*FTLE_dt, c));
        double2 d = Interpolate2DUnsteadyField_device(u, v, step3, w, h, TotalTimeSteps, dx, dy, dt, current_time + FTLE_dt);
        double2 step4 = d2_add(p, d2_smul(FTLE_dt, d));

        if (!is_inbounds_2D_vertex_device(w, h, dx, dy, step1)
        || !is_inbounds_2D_vertex_device(w, h, dx, dy, step2)
        || !is_inbounds_2D_vertex_device(w, h, dx, dy, step3)
        || !is_inbounds_2D_vertex_device(w, h, dx, dy, step4)) 
        return make_d2(-999.0, -999.0);

        double2 incr = d2_smul(FTLE_dt*(1.0/6.0), d2_add(a, d2_add(d2_smul(2.0, b), d2_add(d2_smul(2.0, c), d))));
        double2 p_new = d2_add(p, incr);
        if (!is_inbounds_2D_vertex_device(w, h, dx, dy, p_new)) return make_d2(-999.0, -999.0);
        p = p_new;
    }
    return p;
}

__device__ double2 advect_pathline_2D_rk4_device_tiled(float* u, float* v, const TileDesc& tile, float* tile_u, float* tile_v,
    int w, int h, int TotalTimeSteps, double dx, double dy, double dt, double x0, double y0, double t_i, double FTLE_dt, int FTLE_steps){

    double2 p = make_d2(x0, y0);

    for (int k = 0; k < FTLE_steps; ++k){
        double current_time = t_i + k*FTLE_dt;
        if (current_time < 1e-15 && FTLE_dt < 0) return p;

        double2 a = Interpolate2DUnsteadyField_device_tiled(u, v, tile, tile_u, tile_v, p, w, h, TotalTimeSteps, dx, dy, dt, current_time);
        double2 step1 = d2_add(p, d2_smul(0.5*FTLE_dt, a));
        double2 b = Interpolate2DUnsteadyField_device_tiled(u, v, tile, tile_u, tile_v, step1, w, h, TotalTimeSteps, dx, dy, dt, current_time + FTLE_dt*0.5);
        double2 step2 = d2_add(p, d2_smul(0.5*FTLE_dt, b));
        double2 c = Interpolate2DUnsteadyField_device_tiled(u, v, tile, tile_u, tile_v, step2, w, h, TotalTimeSteps, dx, dy, dt, current_time + FTLE_dt*0.5);
        double2 step3 = d2_add(p, d2_smul(0.5*FTLE_dt, c));
        double2 d = Interpolate2DUnsteadyField_device_tiled(u, v, tile, tile_u, tile_v, step3, w, h, TotalTimeSteps, dx, dy, dt, current_time + FTLE_dt);
        double2 step4 = d2_add(p, d2_smul(FTLE_dt, d));

        if (!is_inbounds_2D_vertex_device(w, h, dx, dy, step1)
        || !is_inbounds_2D_vertex_device(w, h, dx, dy, step2)
        || !is_inbounds_2D_vertex_device(w, h, dx, dy, step3)
        || !is_inbounds_2D_vertex_device(w, h, dx, dy, step4)) 
        return make_d2(-999.0, -999.0);

        double2 incr = d2_smul(FTLE_dt*(1.0/6.0), d2_add(a, d2_add(d2_smul(2.0, b), d2_add(d2_smul(2.0, c), d))));
        double2 p_new = d2_add(p, incr);
        if (!is_inbounds_2D_vertex_device(w, h, dx, dy, p_new)) return make_d2(-999.0, -999.0);
        p = p_new;
    }
    return p;
}

__device__ double FTLE_device(float* u, float* v, int w, int h, int TotalTimeSteps, double dx, double dy, double dt,
    double x0, double y0, double t_i, double FTLE_dt, int FTLE_steps){
    double delx = 0.25*dx;
    double dely = 0.25*dy;
    double2 top = advect_pathline_2D_rk4_device(u, v, w, h, TotalTimeSteps, dx, dy, dt, x0, y0 + dely, t_i, FTLE_dt, FTLE_steps);
    double2 bot = advect_pathline_2D_rk4_device(u, v, w, h, TotalTimeSteps, dx, dy, dt, x0, y0 - dely, t_i, FTLE_dt, FTLE_steps);
    double2 lef = advect_pathline_2D_rk4_device(u, v, w, h, TotalTimeSteps, dx, dy, dt, x0 - delx, y0, t_i, FTLE_dt, FTLE_steps);
    double2 rig = advect_pathline_2D_rk4_device(u, v, w, h, TotalTimeSteps, dx, dy, dt, x0 + delx, y0, t_i, FTLE_dt, FTLE_steps);
    if ((top.x == -999.0 && top.y == -999.0) || (bot.x == -999.0 && bot.y == -999.0) || (lef.x == -999.0 && lef.y == -999.0) || (rig.x == -999.0 && rig.y == -999.0))
        return 0.0;
    double F11 = (rig.x - lef.x) / (2*delx);
    double F12 = (top.x - bot.x) / (2*dely);
    double F21 = (rig.y - lef.y) / (2*delx);
    double F22 = (top.y - bot.y) / (2*dely);
    double a = F11*F11 + F21*F21;
    double b = F11*F12 + F21*F22;
    double d = F12*F12 + F22*F22;
    double trace = a + d;
    double det = a*d - b*b;
    double lambda_max = 0.5*(trace + sqrt(fmax(0.0, trace*trace - 4.0*det)));
    double T = FTLE_steps*fabs(FTLE_dt);
    if (T == 0.0 || lambda_max <= 0.0) return 0.0;
    return (1.0 / fabs(T))*0.5*log(lambda_max);
}

__device__ double FTLE_device_tiled(float* u, float* v, const TileDesc& tile, float* tile_u, float* tile_v,
    int w, int h, int TotalTimeSteps, double dx, double dy, double dt, double x0, double y0, double t_i, double FTLE_dt, int FTLE_steps){

    double delx = 0.25*dx;
    double dely = 0.25*dy;
    double2 top = advect_pathline_2D_rk4_device_tiled(u, v, tile, tile_u, tile_v, w, h, TotalTimeSteps, dx, dy, dt, x0, y0 + dely, t_i, FTLE_dt, FTLE_steps);
    double2 bot = advect_pathline_2D_rk4_device_tiled(u, v, tile, tile_u, tile_v, w, h, TotalTimeSteps, dx, dy, dt, x0, y0 - dely, t_i, FTLE_dt, FTLE_steps);
    double2 lef = advect_pathline_2D_rk4_device_tiled(u, v, tile, tile_u, tile_v, w, h, TotalTimeSteps, dx, dy, dt, x0 - delx, y0, t_i, FTLE_dt, FTLE_steps);
    double2 rig = advect_pathline_2D_rk4_device_tiled(u, v, tile, tile_u, tile_v, w, h, TotalTimeSteps, dx, dy, dt, x0 + delx, y0, t_i, FTLE_dt, FTLE_steps);
    if ((top.x == -999.0 && top.y == -999.0) || (bot.x == -999.0 && bot.y == -999.0) || (lef.x == -999.0 && lef.y == -999.0) || (rig.x == -999.0 && rig.y == -999.0))
        return 0.0;
    double F11 = (rig.x - lef.x) / (2*delx);
    double F12 = (top.x - bot.x) / (2*dely);
    double F21 = (rig.y - lef.y) / (2*delx);
    double F22 = (top.y - bot.y) / (2*dely);
    double a = F11*F11 + F21*F21;
    double b = F11*F12 + F21*F22;
    double d = F12*F12 + F22*F22;
    double trace = a + d;
    double det = a*d - b*b;
    double lambda_max = 0.5*(trace + sqrt(fmax(0.0, trace*trace - 4.0*det)));
    double T = FTLE_steps*fabs(FTLE_dt);
    if (T == 0.0 || lambda_max <= 0.0) return 0.0;
    return (1.0 / fabs(T))*0.5*log(lambda_max);
}


//no need to pass min_x,min_y:
//pos_x=minx+idx*dx, pos_x->float_idx: (pos_x-minx)/dx
//pos_y=miny+idy*dy, pos_y->float_idx: (pos_y-miny)/dy
//this is equivalent to:
//pos_x=idx*dx, pos_x->float_idx: pos_x/dx
//pos_y=idy*dy, pos_y->float_idx: pos_y/dy
//same in time resolution
//the input t_i is relative time instead of physical time, i.e. t_i*v_dt+vector_field.tmin is the physical time
//we don't need to pass min_t or compute physical time, since we can query grid by floorIndex=floor(t_i/dt) ceilIndex=floorIndex+1
__global__ void compute_FTLE_image_kernel(float* field_u, float* field_v, int v_width, int v_height, int TotalTimeSteps, double v_dx, double v_dy, double v_dt,
    double* FTLE_field, int FTLE_size_x, int FTLE_size_y, double FTLE_dx, double FTLE_dy, double t_i, double FTLE_dt, int FTLE_steps){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    //handle the edge case:
    if (ix >= FTLE_size_x || iy >= FTLE_size_y) { return; }
    if (ix < 2 || iy < 2 || ix >= FTLE_size_x-2 || iy >= FTLE_size_y-2) { FTLE_field[ix + iy*FTLE_size_x] = 0.0; return; }

    double x0 = ix*FTLE_dx;
    double y0 = iy*FTLE_dy;
    double ftle = FTLE_device(field_u, field_v, v_width, v_height, TotalTimeSteps, v_dx, v_dy, v_dt, x0, y0, t_i, FTLE_dt, FTLE_steps);
    FTLE_field[ix + iy*FTLE_size_x] = cuda_max(0.0, ftle);
    return;
}

// Shared-memory accelerated variant.
// Callers must provide dynamic shared memory:
// shared_bytes = 2 * tile_w * tile_h * tile_t * sizeof(float)
// and pass tile origin/size (tile_w/tile_h/tile_t).
__global__ void compute_FTLE_image_kernel_tiled(float* field_u, float* field_v, int v_width, int v_height, int TotalTimeSteps, double v_dx, double v_dy, double v_dt,
    double* FTLE_field, int FTLE_size_x, int FTLE_size_y, double FTLE_dx, double FTLE_dy, double t_i, double FTLE_dt, int FTLE_steps,
    int tile_x0, int tile_y0, int tile_t0, int tile_w, int tile_h, int tile_t){

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= FTLE_size_x || iy >= FTLE_size_y) { return; }
    if (ix < 2 || iy < 2 || ix >= FTLE_size_x-2 || iy >= FTLE_size_y-2) { FTLE_field[ix + iy*FTLE_size_x] = 0.0; return; }

    TileDesc tile;
    tile.ox = tile_x0; tile.oy = tile_y0; tile.ot = tile_t0;
    tile.sx = tile_w;  tile.sy = tile_h;  tile.st = tile_t;

    float* tile_u = nullptr;
    float* tile_v = nullptr;

    extern __shared__ float shmem[];
    if (tile_has_data(tile)){
        int tile_slice = tile_w * tile_h;
        int total_elems = tile_slice * tile_t;
        tile_u = shmem;
        tile_v = shmem + total_elems;

        int linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
        int stride = blockDim.x * blockDim.y;
        // Preload the tile (u, v) into shared memory
        for (int idx = linear_idx; idx < total_elems; idx += stride){
            int lt = idx / tile_slice;
            int rem = idx - lt * tile_slice;
            int ly = rem / tile_w;
            int lx = rem - ly * tile_w;

            int gx = tile_x0 + lx;
            int gy = tile_y0 + ly;
            int gt = tile_t0 + lt;
            int g_idx = gt * v_height * v_width + gy * v_width + gx;

            tile_u[idx] = field_u[g_idx];
            tile_v[idx] = field_v[g_idx];
        }
        __syncthreads();
    }

    double x0 = ix*FTLE_dx;
    double y0 = iy*FTLE_dy;
    double ftle = FTLE_device_tiled(field_u, field_v, tile, tile_u, tile_v, v_width, v_height, TotalTimeSteps, v_dx, v_dy, v_dt, x0, y0, t_i, FTLE_dt, FTLE_steps);
    FTLE_field[ix + iy*FTLE_size_x] = cuda_max(0.0, ftle);
    return;
}