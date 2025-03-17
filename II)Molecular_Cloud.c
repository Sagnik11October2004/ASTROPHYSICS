#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <libconfig.h>

/*
 * Quadtree structure for gravity calculations
 */
typedef struct quadtree_node_t {
    double cx, cy;           /* Center coordinates */
    double size;             /* Side length */
    double total_mass;       /* Total mass of particles in this node */
    double com_x, com_y;     /* Center of mass */
    int n_particles;         /* Number of particles in this node */
    int particle_idx;        /* Index of particle if n_particles == 1 */
    struct quadtree_node_t *children[4]; /* NW, NE, SW, SE quadrants */
} quadtree_node_t;

typedef struct sim_param_t {
    char *fname;        /* File name */
    int nframes;        /* Number of frames */
    int npframe;        /* Steps per frame */
    double h;           /* Particle size / smoothing length */
    double dt;          /* Time step */
    double rho0;        /* Reference density */
    double k;           /* Bulk modulus */
    double mu;          /* Viscosity */
    double G;           /* Gravitational constant */
    double gamma;       /* Equation of state exponent */
    double domain_size; /* Size of the simulation domain (parsecs) */
    double theta;       /* Barnes-Hut opening angle parameter */
    int mesh_size;      /* Number of cells in spatial mesh (per dimension) */
} sim_param_t;

int get_params(int argc, char **argv, sim_param_t *params);

typedef struct sim_state_t {
    int n;                   /* Number of particles */
    double mass;             /* Mass per particle */
    double *restrict rho;    /* Density */
    double *restrict x;      /* Position (2D) */
    double *restrict vh;     /* Velocity half-step */
    double *restrict v;      /* Velocity */
    double *restrict a;      /* Acceleration */
    double *restrict p;      /* Pressure at each particle */
    double *restrict a_grav; /* Gravitational acceleration */
    int **mesh;              /* Spatial mesh for neighbor finding */
    int *mesh_counts;        /* Number of particles in each mesh cell */
    int *cell_id;            /* Cell ID for each particle */
} sim_state_t;

sim_state_t *alloc_state(int n, int mesh_size) {
    sim_state_t *ss;
    ss = malloc(sizeof(sim_state_t));
    ss->n = n;
    ss->rho = malloc(sizeof(double) * n);
    ss->x = malloc(sizeof(double) * 2 * n);
    ss->v = malloc(sizeof(double) * 2 * n);
    ss->vh = malloc(sizeof(double) * 2 * n);
    ss->a = malloc(sizeof(double) * 2 * n);
    ss->p = malloc(sizeof(double) * n);
    ss->a_grav = malloc(sizeof(double) * 2 * n);
    ss->cell_id = malloc(sizeof(int) * n);
    
    /* Allocate spatial mesh for neighbor finding */
    ss->mesh_counts = calloc(mesh_size * mesh_size, sizeof(int));
    ss->mesh = malloc(mesh_size * mesh_size * sizeof(int *));
    for (int i = 0; i < mesh_size * mesh_size; i++) {
        ss->mesh[i] = malloc(50 * sizeof(int)); /* Initial capacity */
    }
    
    return ss;
}

void free_state(sim_state_t *s, int mesh_size) {
    free(s->rho);
    free(s->x);
    free(s->v);
    free(s->vh);
    free(s->a);
    free(s->p);
    free(s->a_grav);
    free(s->cell_id);
    
    for (int i = 0; i < mesh_size * mesh_size; i++) {
        free(s->mesh[i]);
    }
    free(s->mesh);
    free(s->mesh_counts);
    free(s);
}

/* Quadtree functions */
quadtree_node_t *create_quadtree_node(double cx, double cy, double size) {
    quadtree_node_t *node = malloc(sizeof(quadtree_node_t));
    node->cx = cx;
    node->cy = cy;
    node->size = size;
    node->total_mass = 0.0;
    node->com_x = 0.0;
    node->com_y = 0.0;
    node->n_particles = 0;
    node->particle_idx = -1;
    
    for (int i = 0; i < 4; i++) {
        node->children[i] = NULL;
    }
    
    return node;
}

void free_quadtree(quadtree_node_t *node) {
    if (!node) return;
    
    for (int i = 0; i < 4; i++) {
        if (node->children[i]) {
            free_quadtree(node->children[i]);
        }
    }
    
    free(node);
}

int get_quadrant(double x, double y, double cx, double cy) {
    if (x < cx) {
        return (y < cy) ? 2 : 0; /* SW : NW */
    } else {
        return (y < cy) ? 3 : 1; /* SE : NE */
    }
}

void insert_particle_to_quadtree(quadtree_node_t *node, int idx, double mass, double *x) {
    if (node->n_particles == 0) {
        /* Empty node, just add the particle */
        node->n_particles = 1;
        node->particle_idx = idx;
        node->total_mass = mass;
        node->com_x = x[2 * idx];
        node->com_y = x[2 * idx + 1];
        return;
    } else if (node->n_particles == 1) {
        /* Split the node */
        int old_idx = node->particle_idx;
        double old_x = node->com_x;
        double old_y = node->com_y;
        double old_mass = node->total_mass;
        
        /* Reset node values */
        node->particle_idx = -1;
        
        /* Create children if they don't exist */
        double half_size = node->size / 2.0;
        double offset = half_size / 2.0;
        
        if (!node->children[0])
            node->children[0] = create_quadtree_node(node->cx - offset, node->cy + offset, half_size); /* NW */
        if (!node->children[1])
            node->children[1] = create_quadtree_node(node->cx + offset, node->cy + offset, half_size); /* NE */
        if (!node->children[2])
            node->children[2] = create_quadtree_node(node->cx - offset, node->cy - offset, half_size); /* SW */
        if (!node->children[3])
            node->children[3] = create_quadtree_node(node->cx + offset, node->cy - offset, half_size); /* SE */
        
        /* Insert the old particle */
        int q_old = get_quadrant(old_x, old_y, node->cx, node->cy);
        insert_particle_to_quadtree(node->children[q_old], old_idx, old_mass, x);
        
        /* Continue with the new particle */
        node->n_particles = 2;
    } else {
        /* Node already has multiple particles */
        node->n_particles++;
    }
    
    /* Insert the new particle into the appropriate quadrant */
    int q = get_quadrant(x[2 * idx], x[2 * idx + 1], node->cx, node->cy);
    insert_particle_to_quadtree(node->children[q], idx, mass, x);
    
    /* Update center of mass and total mass */
    double new_total_mass = node->total_mass + mass;
    node->com_x = (node->total_mass * node->com_x + mass * x[2 * idx]) / new_total_mass;
    node->com_y = (node->total_mass * node->com_y + mass * x[2 * idx + 1]) / new_total_mass;
    node->total_mass = new_total_mass;
}

quadtree_node_t *build_quadtree(sim_state_t *state, double domain_size) {
    /* Create root node that encompasses the entire domain */
    quadtree_node_t *root = create_quadtree_node(domain_size / 2.0, domain_size / 2.0, domain_size);
    
    /* Insert all particles */
    for (int i = 0; i < state->n; i++) {
        insert_particle_to_quadtree(root, i, state->mass, state->x);
    }
    
    return root;
}

/* Calculate gravitational acceleration using Barnes-Hut algorithm */
void calculate_gravity_bh(int i, quadtree_node_t *node, sim_state_t *state, double theta, double G, double softening) {
    if (!node || node->n_particles == 0) return;
    
    double dx = node->com_x - state->x[2 * i];
    double dy = node->com_y - state->x[2 * i + 1];
    double r2 = dx * dx + dy * dy + softening * softening;
    
    /* If this is a leaf node with just one particle that isn't the current particle */
    if (node->n_particles == 1 && node->particle_idx != i) {
        double r = sqrt(r2);
        double f = G * node->total_mass / (r * r * r);
        state->a_grav[2 * i] += f * dx;
        state->a_grav[2 * i + 1] += f * dy;
        return;
    }
    
    /* Check if node is far enough to be approximated */
    double s_over_d = node->size / sqrt(r2);
    if (s_over_d < theta) {
        /* Use center of mass approximation */
        double r = sqrt(r2);
        double f = G * node->total_mass / (r * r * r);
        state->a_grav[2 * i] += f * dx;
        state->a_grav[2 * i + 1] += f * dy;
    } else {
        /* Recurse to children */
        for (int q = 0; q < 4; q++) {
            if (node->children[q]) {
                calculate_gravity_bh(i, node->children[q], state, theta, G, softening);
            }
        }
    }
}

/* Update spatial mesh for neighbor finding */
void update_spatial_mesh(sim_state_t *state, sim_param_t *params) {
    int mesh_size = params->mesh_size;
    double domain_size = params->domain_size;
    
    /* Clear the mesh */
    for (int i = 0; i < mesh_size * mesh_size; i++) {
        state->mesh_counts[i] = 0;
    }
    
    /* Assign particles to mesh cells */
    for (int i = 0; i < state->n; i++) {
        int cell_x = (int)(state->x[2 * i] * mesh_size / domain_size);
        int cell_y = (int)(state->x[2 * i + 1] * mesh_size / domain_size);
        
        /* Ensure we stay within bounds */
        cell_x = (cell_x < 0) ? 0 : (cell_x >= mesh_size ? mesh_size - 1 : cell_x);
        cell_y = (cell_y < 0) ? 0 : (cell_y >= mesh_size ? mesh_size - 1 : cell_y);
        
        int cell_id = cell_y * mesh_size + cell_x;
        state->cell_id[i] = cell_id;
        
        /* Store the particle in the mesh */
        state->mesh[cell_id][state->mesh_counts[cell_id]] = i;
        state->mesh_counts[cell_id]++;
        
        /* Resize if necessary (simple implementation) */
        if (state->mesh_counts[cell_id] % 50 == 0) {
            state->mesh[cell_id] = realloc(state->mesh[cell_id], 
                                         (state->mesh_counts[cell_id] + 50) * sizeof(int));
        }
    }
}

void compute_density_with_mesh(sim_state_t *s, sim_param_t *params) {
    int n = s->n;
    double *restrict rho = s->rho;
    const double *restrict x = s->x;
    double h = params->h;
    double h2 = h * h;
    double h8 = (h2 * h2) * (h2 * h2);
    double C = 4 * s->mass / M_PI / h8;
    int mesh_size = params->mesh_size;
    
    /* Clear densities */
    memset(rho, 0, n * sizeof(double));
    
    for (int i = 0; i < n; ++i) {
        rho[i] += 4 * s->mass / M_PI / h2; /* Self-contribution */
        
        int cell_x = s->cell_id[i] % mesh_size;
        int cell_y = s->cell_id[i] / mesh_size;
        
        /* Check neighboring cells (and own cell) */
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = cell_x + dx;
                int ny = cell_y + dy;
                
                if (nx < 0 || nx >= mesh_size || ny < 0 || ny >= mesh_size) continue;
                
                int neighbor_cell = ny * mesh_size + nx;
                
                /* Check all particles in this cell */
                for (int j = 0; j < s->mesh_counts[neighbor_cell]; j++) {
                    int particle_j = s->mesh[neighbor_cell][j];
                    
                    if (particle_j <= i) continue; /* Avoid double counting */
                    
                    double dx = x[2 * i + 0] - x[2 * particle_j + 0];
                    double dy = x[2 * i + 1] - x[2 * particle_j + 1];
                    double r2 = dx * dx + dy * dy;
                    double z = h2 - r2;
                    
                    if (z > 0) {
                        double rho_ij = C * z * z * z;
                        rho[i] += rho_ij;
                        rho[particle_j] += rho_ij;
                    }
                }
            }
        }
    }
}
void compute_pressure(sim_state_t *state, sim_param_t *params) {
    const double rho0 = params->rho0;
    const double k = params->k;
    const double gamma = params->gamma;
    const double *restrict rho = state->rho;
    double *restrict p = state->p;
    int n = state->n;

    /* For molecular clouds, isothermal equation of state is often used
       Instead of Tait equation, use p = k * rho, where k is related to sound speed */
    if (gamma == 1.0) {
        /* Isothermal case (gamma = 1) */
        for (int i = 0; i < n; ++i) {
            p[i] = k * rho[i];
        }
    } else {
        /* More general polytropic case */
        for (int i = 0; i < n; ++i) {
            p[i] = k * pow(rho[i], gamma);
        }
    }
}

void compute_accel(sim_state_t *state, sim_param_t *params) {
    // Unpack basic parameters
    const double h = params->h;
    const double rho0 = params->rho0;
    const double mu = params->mu;
    const double G = params->G;
    const double theta = params->theta;
    const double mass = state->mass;
    const double h2 = h * h;
    const double domain_size = params->domain_size;
    
    // Unpack system state
    const double *restrict rho = state->rho;
    const double *restrict x = state->x;
    const double *restrict v = state->v;
    double *restrict a = state->a;
    double *restrict a_grav = state->a_grav;
    double *restrict p = state->p;
    int n = state->n;

    // Update the spatial mesh
    update_spatial_mesh(state, params);
    
    // Compute density using the spatial mesh
    compute_density_with_mesh(state, params);
    
    // Compute pressure using appropriate equation of state
    compute_pressure(state, params);
    
    // Initialize acceleration arrays
    for (int i = 0; i < n; ++i) {
        a[2 * i + 0] = 0;
        a[2 * i + 1] = 0;
        a_grav[2 * i + 0] = 0;
        a_grav[2 * i + 1] = 0;
    }
    
    // Constants for SPH interaction term
    double C0 = mass / M_PI / ((h2) * (h2));
    double Cv = -40 * mu;
    
    // Build quadtree for gravity calculations
    quadtree_node_t *root = build_quadtree(state, domain_size);
    
    // Calculate gravity using Barnes-Hut algorithm
    double softening = h * 0.5; // Softening parameter to avoid numerical issues
    for (int i = 0; i < n; ++i) {
        calculate_gravity_bh(i, root, state, theta, G, softening);
    }
    
    // Free the quadtree
    free_quadtree(root);
    
    // Compute SPH interaction forces using the mesh
    for (int i = 0; i < n; ++i) {
        const double rhoi = rho[i];
        const double pi = p[i];
        int cell_x = state->cell_id[i] % params->mesh_size;
        int cell_y = state->cell_id[i] / params->mesh_size;
        
        // Check neighboring cells
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = cell_x + dx;
                int ny = cell_y + dy;
                
                if (nx < 0 || nx >= params->mesh_size || ny < 0 || ny >= params->mesh_size) continue;
                
                int neighbor_cell = ny * params->mesh_size + nx;
                
                // Check all particles in this cell
                for (int jj = 0; jj < state->mesh_counts[neighbor_cell]; jj++) {
                    int j = state->mesh[neighbor_cell][jj];
                    
                    if (j <= i) continue; // Avoid double counting
                    
                    double dx = x[2 * i + 0] - x[2 * j + 0];
                    double dy = x[2 * i + 1] - x[2 * j + 1];
                    double r2 = dx * dx + dy * dy;
                    
                    if (r2 < h2 && r2 > 0) {
                        const double rhoj = rho[j];
                        const double pj = p[j];
                        double q = sqrt(r2) / h;
                        double u = 1 - q;
                        double w0 = C0 * u / rhoi / rhoj;
                        
                        // Pressure force
                        double wp = -w0 * (pi + pj) / (2.0 * q);
                        
                        // Viscosity term
                        double wv = w0 * Cv;
                        
                        double dvx = v[2 * i + 0] - v[2 * j + 0];
                        double dvy = v[2 * i + 1] - v[2 * j + 1];
                        
                        a[2 * i + 0] += (wp * dx + wv * dvx);
                        a[2 * i + 1] += (wp * dy + wv * dvy);
                        a[2 * j + 0] -= (wp * dx + wv * dvx);
                        a[2 * j + 1] -= (wp * dy + wv * dvy);
                    }
                }
            }
        }
        
        // Add gravitational acceleration to hydrodynamic acceleration
        a[2 * i + 0] += a_grav[2 * i + 0];
        a[2 * i + 1] += a_grav[2 * i + 1];
    }
}

/* Periodic boundary conditions for molecular cloud simulations */
static void periodic_bc(sim_state_t* s, double domain_size) {
    double *restrict vh = s->vh;
    double *restrict v = s->v;
    double *restrict x = s->x;
    int n = s->n;
    
    for (int i = 0; i < n; ++i, x += 2, v += 2, vh += 2) {
        // Apply periodic boundary conditions
        if (x[0] < 0)
            x[0] += domain_size;
        else if (x[0] >= domain_size)
            x[0] -= domain_size;
            
        if (x[1] < 0)
            x[1] += domain_size;
        else if (x[1] >= domain_size)
            x[1] -= domain_size;
    }
}

void leapfrog_step(sim_state_t *s, sim_param_t *params) {
    const double *restrict a = s->a;
    double *restrict vh = s->vh;
    double *restrict v = s->v;
    double *restrict x = s->x;
    int n = s->n;
    double dt = params->dt;
    double domain_size = params->domain_size;
    
    for (int i = 0; i < 2 * n; ++i)
        vh[i] += a[i] * dt;
    for (int i = 0; i < 2 * n; ++i)
        v[i] = vh[i] + a[i] * dt / 2;
    for (int i = 0; i < 2 * n; ++i)
        x[i] += vh[i] * dt;
    
    periodic_bc(s, domain_size);
}

void leapfrog_start(sim_state_t *s, sim_param_t *params) {
    const double *restrict a = s->a;
    double *restrict vh = s->vh;
    double *restrict v = s->v;
    double *restrict x = s->x;
    int n = s->n;
    double dt = params->dt;
    double domain_size = params->domain_size;
    
    for (int i = 0; i < 2 * n; ++i)
        vh[i] = v[i] + a[i] * dt / 2;
    for (int i = 0; i < 2 * n; ++i)
        v[i] += a[i] * dt;
    for (int i = 0; i < 2 * n; ++i)
        x[i] += vh[i] * dt;
    
    periodic_bc(s, domain_size);
}

/* Distribution functions for molecular cloud initial conditions */
int molecular_cloud_indicator(double x, double y, double center_x, double center_y, double radius) {
    double dx = (x - center_x);
    double dy = (y - center_y);
    double r2 = dx * dx + dy * dy;
    return (r2 < radius * radius);
}

/* Add turbulent velocity field to particles */
void add_turbulence(sim_state_t *state, double turbulence_strength) {
    // Simple random turbulence model
    srand(time(NULL));
    for (int i = 0; i < state->n; i++) {
        // Random angle
        double angle = 2.0 * M_PI * ((double)rand() / RAND_MAX);
        // Random magnitude (Gaussian-like distribution)
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double mag = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2) * turbulence_strength;
        
        // Add turbulent velocity
        state->v[2 * i + 0] += mag * cos(angle);
        state->v[2 * i + 1] += mag * sin(angle);
    }
}

/* Place particles for molecular cloud simulation */
sim_state_t *place_molecular_cloud(sim_param_t *param) {
    double h = param->h;
    double domain_size = param->domain_size;
    double hh = h / 1.3;
    double cloud_radius = domain_size * 0.25; // 25% of domain size
    double center_x = domain_size * 0.5;
    double center_y = domain_size * 0.5;
    
    // Count mesh points that fall in indicated region
    int count = 0;
    for (double x = 0; x < domain_size; x += hh)
        for (double y = 0; y < domain_size; y += hh)
            count += molecular_cloud_indicator(x, y, center_x, center_y, cloud_radius);
    
    // Populate the particle data structure
    sim_state_t *s = alloc_state(count, param->mesh_size);
    int p = 0;
    for (double x = 0; x < domain_size; x += hh) {
        for (double y = 0; y < domain_size; y += hh) {
            if (molecular_cloud_indicator(x, y, center_x, center_y, cloud_radius)) {
                s->x[2 * p + 0] = x;
                s->x[2 * p + 1] = y;
                s->v[2 * p + 0] = 0;
                s->v[2 * p + 1] = 0;
                ++p;
            }
        }
    }
    
    return s;
}

/* Initialize molecular cloud with appropriate mass scaling */
sim_state_t *init_molecular_cloud(sim_param_t *param) {
    sim_state_t *s = place_molecular_cloud(param);
    
    // Set particle mass based on desired cloud mass
    // For molecular clouds, typical masses are in solar masses
    // Here we normalize to achieve the desired reference density
    s->mass = 1.0;
    
    // Compute initial density distribution
    update_spatial_mesh(s, param);
    compute_density_with_mesh(s, param);
    
    // Normalize mass to achieve desired average density
    double rho0 = param->rho0;
    double total_rho = 0.0;
    for (int i = 0; i < s->n; ++i) {
        total_rho += s->rho[i];
    }
    s->mass *= (rho0 * s->n / total_rho);
    
    // Add turbulent initial velocities (characteristic of molecular clouds)
    add_turbulence(s, 0.1); // 10% of sound speed
    
    return s;
}

void check_state(sim_state_t *s, double domain_size) {
    for (int i = 0; i < s->n; ++i) {
        double xi = s->x[2 * i + 0];
        double yi = s->x[2 * i + 1];
        assert(xi >= 0 && xi <= domain_size);
        assert(yi >= 0 && yi <= domain_size);
    }
}

void write_header(FILE *fp, int n);
void write_frame_data(FILE *fp, int n, double *x, int *c);

int main(int argc, char **argv) {
    sim_param_t params;
    if (get_params(argc, argv, &params) != 0)
        exit(-1);
    
    sim_state_t *state = init_molecular_cloud(&params);
    FILE *fp = fopen(params.fname, "wb");
    int nframes = params.nframes;
    int npframe = params.npframe;
    double dt = params.dt;
    int n = state->n;
    
    write_header(fp, n);
    write_frame_data(fp, n, state->x, NULL);
    
    compute_accel(state, &params);
    leapfrog_start(state, &params);
    check_state(state, params.domain_size);
    
    for (int frame = 1; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            compute_accel(state, &params);
            leapfrog_step(state, &params);
            check_state(state, params.domain_size);
            printf("%d/%d-%d/%d\r", frame+1, nframes, i+1, npframe);
            fflush(stdout);
        }
        write_frame_data(fp, n, state->x, NULL);
    }
    
    printf("\nOutput written to: %s\n", params.fname);
    fclose(fp);
    free_state(state, params.mesh_size);
}

static void default_params(sim_param_t *params) {
    params->fname = "molecular_cloud.out";
    params->nframes = 500;
    params->npframe = 20;
    params->dt = 1e-3;         // Larger timestep for astronomical scales
    params->h = 0.01;          // Particle size relative to domain
    params->rho0 = 1.0;        // Normalized density
    params->k = 0.1;           // Related to sound speed squared
    params->mu = 0.01;         // Low viscosity for molecular clouds
    params->G = 1.0;           // Gravitational constant (in simulation units)
    params->gamma = 1.0;       // Isothermal equation of state (gamma = 1)
    params->domain_size = 10.0; // Domain size in parsecs
    params->theta = 0.5;       // Barnes-Hut opening angle parameter
    params->mesh_size = 50;    // Spatial mesh resolution
}

static void read_conf_file(sim_param_t *params, char* confFile) {
    config_t conf, *c;
    const char *ofile;
    c = &conf;
    config_init(c);
    config_read_file(c, confFile);
    
    config_lookup_string(c, "simpara.output", &ofile);
    strcpy(params->fname = malloc(sizeof(char) * (strlen(ofile) + 1)), ofile);
    config_lookup_int(c, "simpara.num_frames", &params->nframes);
    config_lookup_int(c, "simpara.num_per_frames", &params->npframe);
    config_lookup_float(c, "simpara.particle.size", &params->h);
    config_lookup_float(c, "simpara.timestep", &params->dt);
    config_lookup_float(c, "simpara.constants.rho", &params->rho0);
    config_lookup_float(c, "simpara.constants.k", &params->k);
    config_lookup_float(c, "simpara.constants.mu", &params->mu);
    config_lookup_float(c, "simpara.constants.G", &params->G);
    config_lookup_float(c, "simpara.constants.gamma", &params->gamma);
    config_lookup_float(c, "simpara.constants.domain_size", &params->domain_size);
    config_lookup_float(c, "simpara.constants.theta", &params->theta);
    config_lookup_int(c, "simpara.constants.mesh_size", &params->mesh_size);
    
    config_destroy(c);
}

static void print_usage() {
    sim_param_t param;
    default_params(&param);
    fprintf(stderr,
            "Molecular Cloud Dynamics Simulation\n"
            "\t-h: print this message\n"
            "\t-c: config file name\n"
            "\t-o: output file name (%s)\n"
            "\t-F: number of frames (%d)\n"
            "\t-f: steps per frame (%d)\n"
            "\t-t: time step (%e)\n"
            "\t-s: particle size (%e)\n"
            "\t-d: reference density (%g)\n"
            "\t-k: sound speed parameter (%g)\n"
            "\t-v: dynamic viscosity (%g)\n"
            "\t-g: gravitational constant (%g)\n"
            "\t-e: EOS exponent gamma (%g)\n"
            "\t-D: domain size in parsecs (%g)\n"
            "\t-T: Barnes-Hut opening angle (%g)\n"
            "\t-M: spatial mesh resolution (%d)\n",
            param.fname, param.nframes, param.npframe, param.dt, param.h,
            param.rho0, param.k, param.mu, param.G, param.gamma,
            param.domain_size, param.theta, param.mesh_size);
}

int get_params(int argc, char **argv, sim_param_t *params) {
    extern char *optarg;
    const char *optstring = "ho:F:f:t:s:d:k:v:g:e:c:D:T:M:";
    int c;

#define get_int_arg(c, field)                                                  \
    case c:                                                                     \
        params->field = atoi(optarg);                                           \
        break
#define get_flt_arg(c, field)                                                  \
    case c:                                                                     \
        params->field = (double)atof(optarg);                                   \
        break

    default_params(params);
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
            case 'h':
                print_usage();
                return -1;
            case 'o':
                strcpy(params->fname = malloc(strlen(optarg) + 1), optarg);
                break;
            case 'c':
                read_conf_file(params, optarg);
                break;
                get_int_arg('F', nframes);
                get_int_arg('f', npframe);
                get_flt_arg('t', dt);
                get_flt_arg('s', h);
                get_flt_arg('d', rho0);
                get_flt_arg('k', k);
                get_flt_arg('v', mu);
                get_flt_arg('g', G);
                get_flt_arg('e', gamma);
                get_flt_arg('D', domain_size);
                get_flt_arg('T', theta);
                get_int_arg('M', mesh_size);
            default:
                fprintf(stderr, "Unknown option\n");
                return -1;
        }
    }
    return 0;
}

/* Write simulation data to file */
void write_header(FILE *fp, int n) {
    fwrite(&n, sizeof(n), 1, fp);
}

void write_frame_data(FILE *fp, int n, double *x, int *c) {
    for (int i = 0; i < n; ++i) {
        double xi = *(x++);
        double yi = *(x++);
        fwrite(&xi, sizeof(xi), 1, fp);
        fwrite(&yi, sizeof(yi), 1, fp);
        /* In molecular cloud simulations, we might want to add more data fields
           such as density, temperature, or velocity magnitude */
        double ci0 = c ? *c++ : 0;
        fwrite(&ci0, sizeof(ci0), 1, fp);
    }
}

/* Sample config file: molecular_cloud.conf
{
    simpara = {
        output = "molecular_cloud.out";
        num_frames = 500;
        num_per_frames = 20;
        particle = {
            size = 0.01;
        };
        timestep = 1.0e-3;
        constants = {
            rho = 1.0;
            k = 0.1;
            mu = 0.01;
            G = 1.0;
            gamma = 1.0;
            domain_size = 10.0;
            theta = 0.5;
            mesh_size = 50;
        };
    };
}
*/
