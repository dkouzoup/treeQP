
/* Dimensions */

int Nn = 6;
int nk[6] = { 2, 2, 1, 0, 0, 0, };
int nx[6] = { 2, 3, 2, 1, 1, 1, };
int nu[6] = { 1, 2, 1, 0, 0, 0, };

/* Data */

double A[18] = { 3.195997351804961e-01, 5.308642806941265e-01, 6.544457077570663e-01, 4.076191970411526e-01, 8.199812227819406e-01, 7.183589432058837e-01, 9.159912441314253e-01, 1.151057129107236e-03, 4.624491592423287e-01, 4.243490398153752e-01, 7.655000166214384e-01, 1.886619767914910e-01, 2.874981730661312e-01, 6.476176301726844e-01, 6.790167540932019e-01, 6.357867105140836e-01, 4.501376969658960e-01, 4.587254936488678e-01, };
double B[10] = { 9.686493302310937e-01, 5.313339065656745e-01, 3.251456818205600e-01, 4.609163660289640e-01, 7.701597286086093e-01, 9.111346368653495e-02, 5.762093806630070e-01, 9.451741131094014e-01, 2.089349224260229e-01, 6.619447519056519e-01, };
double b[8] = { 1.056292033290219e-01, 6.109586587462006e-01, 7.788022418240925e-01, 3.224718071867786e-01, 7.847392947607416e-01, 6.833632432946530e-01, 7.092817027105448e-01, 7.702855148036601e-01, };
double Q[20] = { 4.654116211214980e+00, 2.371018155661828e-01, 2.371018155661828e-01, 1.121059575629715e+00, 5.603974002573779e+00, 1.222400016893730e-01, 3.968071162698623e-01, 1.222400016893730e-01, 9.717231548017748e+00, 4.487547523446974e-01, 3.968071162698623e-01, 4.487547523446974e-01, 7.252462579325925e+00, 5.206217083363815e+00, 1.058185744763244e-01, 1.058185744763244e-01, 2.248970037773420e+00, 4.803881533302201e+00, 1.430193054966852e+00, 6.970313997032450e+00, };
double R[6] = { 1.540372181473409e+00, 2.301310270670013e+00, 4.584146696719560e-01, 4.584146696719560e-01, 1.679345570777951e+00, 1.555903034585804e+00, };
double S[10] = { 6.959493133016079e-01, 6.998878499282916e-01, 6.799276847001057e-02, 2.547901565970053e-01, 2.240400308242187e-01, 6.678327270137165e-01, 8.443921565272046e-01, 3.444624113010422e-01, 1.917452554617978e-01, 7.384268399769416e-01, };
double q[10] = { 6.385307582718379e-01, 3.360383606642947e-02, 7.805196527313579e-01, 6.753320657469996e-01, 6.715314318477494e-03, 2.428495983181691e-01, 9.174243420493824e-01, 6.444427814313365e-01, 6.073039406856346e-01, 4.161585899697965e-01, };
double r[4] = { 6.880609911805124e-02, 6.021704875817953e-01, 3.867711945209846e-01, 2.690615866860183e-01, };

/* Optimal solution */

double xopt[10] = { -1.087803657972126e-01, -6.619837007079243e-01, -4.235501225456333e-01, -1.127899264391803e-01, 1.566862937671087e-01, -1.901647217099338e-01, 3.251447442418065e-01, 8.246926352530137e-02, -6.777704953200514e-02, 1.739931285764129e-01, };
double uopt[4] = { -2.318443612432541e-01, -4.570094777307827e-01, -4.491327265231564e-01, -9.968267861670170e-01, };
