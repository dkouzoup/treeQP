
/* Dimensions */

int Nn = 5;
int nk[5] = { 2, 1, 1, 0, 0, };
int nx[5] = { 2, 2, 2, 1, 4, };
int nu[5] = { 2, 2, 2, 0, 0, };

/* Data */

double A[18] = { 2.186766323996339e-01, 1.057982732502282e-01, 1.096974645231942e-01, 6.359137097510570e-02, 2.427853578209619e-01, 4.424023130019433e-01, 6.877960851201071e-01, 3.592282104018606e-01, 7.688542524296149e-01, 1.672535454947217e-01, 1.998628228574520e-01, 4.069548371389068e-01, 7.487057182156914e-01, 8.255838157861556e-01, 7.899630299445308e-01, 3.185242453989922e-01, 5.340641273707264e-01, 8.995067877058105e-02, };
double B[18] = { 4.045799958576258e-01, 4.483729120664953e-01, 3.658161768381712e-01, 7.635046408488134e-01, 7.363400743012017e-01, 3.947074752787632e-01, 6.834158669679784e-01, 7.040474303342660e-01, 8.619804787020722e-01, 9.898721536315039e-01, 1.117057441932034e-01, 1.362925489382987e-01, 6.786523048001882e-01, 4.951770190896606e-01, 1.897104060175802e-01, 4.950058249902208e-01, 1.476082219766887e-01, 5.497414690618818e-02, };
double b[9] = { 6.278963796141688e-01, 7.719803855542451e-01, 4.423054133833708e-01, 1.957762355331871e-02, 5.144234565057044e-01, 8.507126742890074e-01, 5.605595273548848e-01, 9.296088667566632e-01, 6.966672005552276e-01, };
double Q[29] = { 6.664420797963579e+00, 5.446789056521815e-01, 5.446789056521815e-01, 6.020854108056972e+00, 7.895516941108768e+00, 5.823846017153944e-01, 5.823846017153944e-01, 1.077074470577811e+00, 8.548069729827169e+00, 3.472899601326014e-01, 3.472899601326014e-01, 4.496267891928119e+00, 6.764541576211930e+00, 7.969193885129858e+00, 4.079597934171830e-01, 7.033469869678101e-01, 7.435029536379893e-01, 4.079597934171830e-01, 6.725308949277783e+00, 5.460449273471087e-01, 9.454013471204179e-01, 7.033469869678101e-01, 5.460449273471087e-01, 3.268692865381701e+00, 4.012520774181147e-01, 7.435029536379893e-01, 9.454013471204179e-01, 4.012520774181147e-01, 7.509340358248717e+00, };
double R[12] = { 2.749599973047848e+00, 2.914265509783446e-01, 2.914265509783446e-01, 1.410318290999238e+00, 1.312317126289869e+00, 6.957420148930979e-01, 6.957420148930979e-01, 1.827715612900154e+00, 2.504799146031044e+00, 5.801486914247295e-01, 5.801486914247295e-01, 1.906945715027784e+00, };
double S[12] = { 4.794632249488878e-01, 6.393169610401084e-01, 5.447161105267628e-01, 6.473114802931277e-01, 7.412579434542065e-01, 5.200524673903868e-01, 3.477126712775251e-01, 1.499972538316831e-01, 3.773955448351028e-01, 2.160189159613943e-01, 7.904072179669134e-01, 9.493039118497969e-01, };
double q[11] = { 5.438859339996390e-01, 7.210466205798114e-01, 5.860920672314619e-01, 2.621453177278074e-01, 3.275654340752052e-01, 6.712643704517398e-01, 1.547523486560447e-01, 8.348281360262266e-02, 6.259597851715832e-01, 6.609445579473423e-01, 7.297518553172211e-01, };
double r[6] = { 5.224953057771021e-01, 9.937046241208520e-01, 4.445409227823849e-02, 7.549332672311792e-01, 4.386449825869556e-01, 8.335005955889753e-01, };

/* Optimal solution */

double xopt[11] = { 1.694584957028678e-02, 3.181263084397845e-02, 2.211107091053585e-01, 3.097828839394523e-02, -2.992290652680776e-01, -6.426697711328722e-01, 2.455222980059268e-02, 1.588651877427041e-01, -1.377479692906447e-02, -2.660912738115484e-02, 1.272513355379190e-01, };
double uopt[6] = { -3.010149503436867e-01, -7.987521818520265e-01, -3.997429609575054e-02, -6.370492381545425e-01, -4.937963721608030e-01, -3.647531595309952e-01, };
