
/* Dimensions */

int Nn = 6;
int nc[6] = { 2, 2, 1, 0, 0, 0, };
int nx[6] = { 2, 3, 2, 1, 1, 1, };
int nu[6] = { 1, 2, 1, 0, 0, 0, };

/* Data */

double A[18] = { 4.795233852102188e-01, 8.013476055219523e-01, 2.278429357060419e-01, 4.980942911963897e-01, 9.008524885320048e-01, 5.746612191301876e-01, 1.208595710985582e-01, 8.627107186996696e-01, 4.842965112121025e-01, 8.448556745762632e-01, 2.054941709076799e-01, 1.465149106148899e-01, 1.890721744726138e-01, 4.991160134825895e-01, 5.358010557511133e-01, 4.451831652960423e-01, 2.084613587513138e-01, 5.649795707382012e-01, };
double B[10] = { 8.451781850540366e-01, 7.386402919954018e-01, 5.859870358264758e-01, 2.094050840209347e-01, 5.522913415387750e-01, 4.265241091114336e-02, 6.351979168598820e-01, 1.239322775980701e-01, 4.903572934680182e-01, 6.403118251627581e-01, };
double b[8] = { 2.467345259859749e-01, 6.664162173194680e-01, 8.348281360262266e-02, 6.298833850644212e-01, 3.199101576256691e-02, 2.818668558804299e-01, 8.529981553408162e-01, 4.170289516428860e-01, };
double Qd[10] = { 9.889116160795892e+00, 5.223753569447709e-03, 6.259597851715832e+00, 6.609445579473423e+00, 7.297518553172210e+00, 6.147134191171405e+00, 3.624114622730526e+00, 5.385966780453404e+00, 8.739274058617333e+00, 2.059755155322434e+00, };
double Rd[4] = { 8.654385910130246e-01, 8.907521163253223e-01, 9.823032228836064e-01, 4.953257904206121e-02, };
double q[10] = { 6.125664694839987e-01, 9.899502057088309e-01, 7.690290853358962e-01, 5.814464878753978e-01, 9.283130623141880e-01, 4.895699891773218e-01, 1.925103960620748e-01, 6.951630394443320e-01, 2.702943322926976e-01, 9.479331212931686e-01, };
double r[4] = { 5.276800693384424e-01, 5.800903657584416e-01, 1.698293833726128e-02, 1.230837475459452e-01, };

/* Optimal solution */

double xopt[10] = { 6.051108326644172e-02, -9.853421760293388e-01, -1.314583401859557e-01, -9.969342905487188e-02, -4.110168629054822e-01, 1.807080836806778e-01, -6.936584546424864e-01, -2.272836851603157e-01, 1.707802208835882e-01, -5.193931170282526e-01, };
double uopt[4] = { 9.889508339274711e-02, -8.711678854284242e-01, -5.551982658814718e-01, -9.092286640836886e-01, };