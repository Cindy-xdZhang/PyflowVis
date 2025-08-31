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



__device__ bool valid_space_time_check_device(double2 p_new, double t_new, double t_target, double dt_local, int v_width, int v_height, double v_dx, double v_dy){
    bool in_bounds = is_inbounds_2D_vertex_device(v_width, v_height, v_dx, v_dy, p_new);
    bool time_ok = (dt_local > 0.0) ? (t_new <= t_target) : (t_new >= t_target);
    return in_bounds && time_ok;
}



// ----------------------- Batched Pathline Integration Kernel -----------------------
//no need to pass min_x,min_y:
//pos_x=minx+idx*dx, pos_x->float_idx: (pos_x-minx)/dx
//pos_y=miny+idy*dy, pos_y->float_idx: (pos_y-miny)/dy
//this is equivalent to:
//pos_x=idx*dx, pos_x->float_idx: pos_x/dx
//pos_y=idy*dy, pos_y->float_idx: pos_y/dy
//same in time resolution
//the input t_i is relative time instead of physical time, i.e. t_i*v_dt+vector_field.tmin is the physical time
//we don't need to pass min_t or compute physical time, since we can query grid by floorIndex=floor(t_i/dt) ceilIndex=floorIndex+1
__global__ void integrate_pathlines_2d_kernel(
    float* field_u, float* field_v,
    int v_width, int v_height, int TotalTimeSteps,
    double v_dx, double v_dy, double v_dt,
    const double* seeds_x, const double* seeds_y,
    const double* t_starts,
    double t_target,
    double step_dt,
    int max_steps,
    int num_seeds,
    double t_min_phys,
    double x_min_phys, double y_min_phys,
    float* out_positions, // shape [num_seeds, max_steps, 3]
    int* out_valid_steps,//shape [num_seeds] the length of the pathline, if the seeding is out of the domain/at critical point, the length is 1(pathline is the seeding itself)
    int methodId//0: Euler, 1: RK4
  ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_seeds) return;

    // Initialize state per seed
    double2 p = make_d2(seeds_x[i], seeds_y[i]); // zero-based coordinates
    double t0 = t_starts[i];                     // relative time (t - tmin)
    double dt_local = (t_target >= t0) ? fabs(step_dt) : -fabs(step_dt);
    double t = t0;

    // Output base index (max_steps samples: index 0..max_steps-1)
    int stride = (max_steps) * 3;
    int base = i * stride;

    // Write step 0 in physical units
    out_positions[base + 0] = (float)(p.x + x_min_phys);
    out_positions[base + 1] = (float)(p.y + y_min_phys);
    out_positions[base + 2] = (float)(t + t_min_phys);

    // Early stop if velocity ~ 0 at start
    double2 v0 = Interpolate2DUnsteadyField_device(field_u, field_v, p, v_width, v_height, TotalTimeSteps, v_dx, v_dy, v_dt, t);

    if (fabs(v0.x) < 1e-9 && fabs(v0.y) < 1e-9 || !valid_space_time_check_device(p, t, t_target, dt_local, v_width, v_height, v_dx, v_dy)){
        out_valid_steps[i] = 1;
        return;
    }

    int pathlineLength = 1;

    if(methodId==0){

     for (int k = 0; k < max_steps-1; ++k){
        // Euler integrate one step
        double2 k1 = Interpolate2DUnsteadyField_device(field_u, field_v, p, v_width, v_height, TotalTimeSteps, v_dx, v_dy, v_dt, t);
        double2 incr =d2_smul(dt_local, k1);
        double2 p_new = d2_add(p, incr);
        double t_new = t + dt_local;

        // Termination checks
        if (!valid_space_time_check_device(p_new, t_new, t_target, dt_local, v_width, v_height, v_dx, v_dy)){
            break;
        }

        //pass the boundary and time termination checks
        // Write step k+1 in physical units
        int out_idx = base + (k+1)*3;
        out_positions[out_idx + 0] = (float)(p_new.x + x_min_phys);
        out_positions[out_idx + 1] = (float)(p_new.y + y_min_phys);
        out_positions[out_idx + 2] = (float)(t_new + t_min_phys);
        pathlineLength++;
        p = p_new;
        t = t_new;
    }
    }
    else{
    for (int k = 0; k < max_steps-1; ++k){
        // RK4 integrate one step
        double2 k1 = Interpolate2DUnsteadyField_device(field_u, field_v, p, v_width, v_height, TotalTimeSteps, v_dx, v_dy, v_dt, t);
        double2 p2 = d2_add(p, d2_smul(0.5*dt_local, k1));
        double2 k2 = Interpolate2DUnsteadyField_device(field_u, field_v, p2, v_width, v_height, TotalTimeSteps, v_dx, v_dy, v_dt, t + 0.5*dt_local);
        double2 p3 = d2_add(p, d2_smul(0.5*dt_local, k2));
        double2 k3 = Interpolate2DUnsteadyField_device(field_u, field_v, p3, v_width, v_height, TotalTimeSteps, v_dx, v_dy, v_dt, t + 0.5*dt_local);
        double2 p4 = d2_add(p, d2_smul(dt_local, k3));
        double2 k4 = Interpolate2DUnsteadyField_device(field_u, field_v, p4, v_width, v_height, TotalTimeSteps, v_dx, v_dy, v_dt, t + dt_local);
        double2 incr = d2_smul((1.0/6.0)*dt_local, d2_add(k1, d2_add(d2_smul(2.0, k2), d2_add(d2_smul(2.0, k3), k4))));
        double2 p_new = d2_add(p, incr);
        double t_new = t + dt_local;

        // Termination checks
        // bool in_bounds = is_inbounds_2D_vertex_device(v_width, v_height, v_dx, v_dy, p_new);
        // bool time_ok = (dt_local > 0.0) ? (t_new < t_target) : (t_new > t_target);
        // if ((!in_bounds) || (!time_ok)){
        //     break;
        // }
        if (!valid_space_time_check_device(p_new, t_new, t_target, dt_local, v_width, v_height, v_dx, v_dy)){
            break;
        }


        //pass the boundary and time termination checks
        // Write step k+1 in physical units
        int out_idx = base + (k+1)*3;
        out_positions[out_idx + 0] = (float)(p_new.x + x_min_phys);
        out_positions[out_idx + 1] = (float)(p_new.y + y_min_phys);
        out_positions[out_idx + 2] = (float)(t_new + t_min_phys);
        pathlineLength++;
        p = p_new;
        t = t_new;
    }
    }

    out_valid_steps[i] = pathlineLength; // include step 0
}