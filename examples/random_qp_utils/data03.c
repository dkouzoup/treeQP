
/* Dimensions */

int Nn = 14;
int nk[14] = { 3, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, };
int nx[14] = { 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, };
int nu[14] = { 2, 2, 2, 3, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, };

/* Data */

double A[62] = { 8.333635634531343e-01, 4.036286627736071e-01, 3.901759381306066e-01, 3.604488933789446e-01, 1.402553592694443e-01, 2.601301941124401e-01, 5.237800369658695e-01, 2.648725935485898e-01, 6.835722047082915e-02, 4.363270774801027e-01, 1.738530373650010e-01, 2.610710815490491e-02, 5.324264825001326e-01, 7.164973465043818e-01, 1.793018438217894e-01, 3.365329258010625e-01, 1.877129485752673e-01, 3.219271831829238e-01, 4.315111829018908e-01, 6.944039096845543e-01, 2.567845632701479e-01, 9.758649885159509e-03, 3.416441046451417e-01, 9.328978958180805e-01, 3.906675366175962e-01, 2.732167079999634e-01, 6.002621455076278e-01, 1.726045016476686e-01, 9.034673814322458e-02, 2.552622026435888e-01, 9.726510769774971e-01, 7.104086852781695e-01, 3.118599451475329e-01, 2.914571276477274e-01, 5.521750267158347e-01, 3.293989274987663e-02, 5.386292643555612e-02, 8.050632285589021e-01, 4.513748547034482e-01, 3.826462295599585e-01, 7.636733236637612e-01, 5.588205505095603e-01, 1.838429444657746e-01, 4.979488150189469e-01, 9.624314043666340e-01, 5.350671052561570e-01, 9.638701299717145e-01, 1.156258791812592e-01, 3.690916888248605e-01, 6.617649133654143e-01, 1.696088134543144e-01, 2.787840204390187e-01, 4.649542833145343e-01, 8.139769268209339e-01, 8.984441371801204e-01, 4.292385431148273e-01, 9.332512027769413e-03, 9.150259117357898e-01, 6.427417391331040e-01, 1.419058202402956e-03, 3.038527249491774e-02, 2.084702233053197e-01, };
double B[55] = { 8.681510086583366e-02, 4.293973370858053e-01, 2.572827847698599e-01, 2.975553841511184e-01, 9.546782740804495e-01, 4.305965198594173e-01, 9.615585731036632e-01, 7.624144840029929e-01, 4.038567112302067e-01, 5.485662998569636e-01, 4.873859278226667e-02, 5.527321331789425e-01, 5.322830710608887e-01, 2.793919652529646e-01, 9.462301535199267e-01, 9.064432665100022e-01, 1.519470798468442e-01, 3.971088427434517e-01, 3.747224669512426e-01, 1.311147070430053e-01, 8.585705312592099e-01, 9.110670533977940e-01, 6.996337672257625e-01, 7.251823550233875e-01, 8.503573373746214e-01, 9.116474240078534e-01, 6.392761472760637e-01, 2.553702979444434e-01, 7.896437036896907e-01, 3.642868694997939e-01, 5.323499349989104e-01, 7.116567059812665e-01, 8.714765179958467e-01, 3.286896116722290e-01, 6.501180253977771e-01, 9.748361480027575e-01, 7.596736129413562e-02, 5.178456002340507e-01, 9.942430106440503e-01, 8.548516830906751e-01, 9.624039397119708e-01, 5.144829323020528e-02, 3.043489456365734e-01, 5.801918331427124e-01, 5.309644523382815e-01, 1.982217945138389e-01, 1.950715332842609e-01, 3.268396483499757e-01, 8.803378603791996e-01, 3.343294196222197e-01, 5.966471044452463e-01, 9.019908087065621e-01, 7.020664493596758e-01, 4.549661450020972e-01, 1.272660333251334e-01, };
double b[27] = { 4.248584117046258e-01, 1.192072594212871e-01, 7.348661102846932e-03, 6.800386404601387e-01, 2.748114048375007e-01, 2.415017417341122e-01, 3.926845759741115e-01, 2.485523384487209e-02, 4.350407178956268e-01, 9.151316721261082e-02, 2.298860788574965e-01, 5.760534563213544e-01, 8.866584003228306e-02, 8.382555875372257e-01, 5.870191670827718e-01, 4.138864977733601e-01, 3.091364264662670e-01, 6.789410089770332e-01, 4.035013888043607e-01, 9.012080926531435e-01, 5.405504251702443e-01, 4.711018650157485e-01, 4.039693721709411e-01, 3.774551020708649e-01, 7.349559254300144e-01, 8.647687878090338e-03, 7.270796011485272e-01, };
double Q[66] = { 2.215613277053294e+00, 3.896352276235258e-01, 9.185658051588974e-02, 3.896352276235258e-01, 9.971187769020304e+00, 7.122416444967276e-01, 9.185658051588974e-02, 7.122416444967276e-01, 6.421810786131930e+00, 1.235962692409728e+00, 4.749903001092556e-01, 4.749903001092556e-01, 4.723904351750972e+00, 8.429612890379310e+00, 5.987193155575562e-01, 5.987193155575562e-01, 2.498391965787782e+00, 8.430289551488988e+00, 5.552879050890492e-01, 5.552879050890492e-01, 8.218279853744940e+00, 5.174674923906990e+00, 9.043351368441590e-01, 9.043351368441590e-01, 5.881635902798120e+00, 2.968294689521640e+00, 2.921197377770920e-01, 2.921197377770920e-01, 5.269926992666271e+00, 4.020038431485553e+00, 6.961413177919054e-01, 6.961413177919054e-01, 5.204088203178667e+00, 3.435799475849739e+00, 5.045688322942656e-01, 5.045688322942656e-01, 8.861963037837743e+00, 7.552472722040535e+00, 4.726688546107866e-01, 9.937874771548392e-01, 4.726688546107866e-01, 5.764681093770282e+00, 4.990297753955104e-01, 9.937874771548392e-01, 4.990297753955104e-01, 8.514129381893754e+00, 7.985753837305940e+00, 3.556380786477992e-01, 3.556380786477992e-01, 5.981880594115276e+00, 8.441189431435165e+00, 6.275408966129317e-01, 6.275408966129317e-01, 1.441767962178138e+00, 6.332482447731942e+00, 6.881903669749561e-01, 6.881903669749561e-01, 4.610598176810820e+00, 1.666448505717051e+00, 5.414594727314586e-01, 5.414594727314586e-01, 2.130914311294217e+00, 8.462482696214501e-01, 6.085512938656579e-01, 6.085512938656579e-01, 9.328737975661705e-01, };
double R[38] = { 2.120029522200178e+00, 7.989163752842121e-01, 7.989163752842121e-01, 1.517192714209078e+00, 2.883622326062504e-01, 1.109887303168101e-01, 1.109887303168101e-01, 7.253316692414872e-01, 1.006902953966875e+00, 8.736528643769779e-01, 8.736528643769779e-01, 1.619704411083423e+00, 7.506662632406150e-01, 1.807846869412485e-01, 4.070225764103227e-01, 1.807846869412485e-01, 1.319737727225923e+00, 6.213395808704536e-01, 4.070225764103227e-01, 6.213395808704536e-01, 1.962129574726963e+00, 1.434333146176267e+00, 6.847367522606944e-01, 6.847367522606944e-01, 1.890078717338384e+00, 1.011889643539141e+00, 2.788451639653659e-01, 2.788451639653659e-01, 2.316428125649060e+00, 1.720571230634633e+00, 6.411218360539179e-01, 6.411218360539179e-01, 2.246877970624864e+00, 1.544728627574236e+00, 4.179707342852300e-01, 4.179707342852300e-01, 1.241780047802726e+00, 5.029958024606609e-01, };
double S[39] = { 5.538870657912750e-01, 6.800655300833607e-01, 3.671899053173674e-01, 2.392906061935454e-01, 5.789234924590936e-01, 8.668870546725078e-01, 1.748920653357828e-01, 1.386489715925953e-01, 5.988856103678650e-01, 9.010579056827256e-01, 9.101952266944817e-01, 9.090981878008823e-01, 5.915944089074379e-01, 3.325714073355511e-01, 7.624080487887140e-01, 5.760559014965571e-01, 7.476628376417164e-01, 6.455345059813216e-01, 1.232195183245062e-01, 5.043978600927667e-01, 1.161185127794003e-01, 5.765436121487222e-02, 9.797652239759865e-01, 2.848237268606177e-01, 9.710752314425135e-01, 8.463728876931734e-01, 5.059994558905483e-01, 2.788756111452306e-01, 7.890289233139490e-01, 3.178330537262285e-01, 4.522074537629818e-01, 7.522279700499424e-01, 3.181047261502632e-01, 1.338109853561260e-01, 6.714628894780307e-01, 5.709910754624057e-01, 5.909914552748493e-01, 9.101878307281412e-01, 1.937659361666103e-01, };
double q[30] = { 4.067767602152256e-01, 1.126151410250468e-01, 4.438458367269573e-01, 9.393797658842382e-01, 2.211844559390779e-01, 8.530636292099192e-01, 4.423978930411651e-01, 3.472613127216336e-01, 9.214768480510438e-02, 5.949742988095963e-01, 9.621610310112598e-01, 7.466172218282013e-01, 2.369303841128702e-01, 1.098617057506858e-01, 1.097423685939036e-01, 1.697670660264885e-01, 1.476557771517368e-01, 4.323677915343976e-01, 7.491597290680092e-01, 3.918448664758301e-02, 7.566307008943676e-01, 9.954810585552578e-01, 4.784744729028632e-01, 2.568353541796335e-01, 8.771817493370971e-01, 7.848524272830237e-01, 9.298892687067795e-02, 4.634892477622432e-01, 9.110017541305393e-02, 5.940370314441212e-01, };
double r[18] = { 3.001844012139000e-01, 4.013868538144928e-01, 4.826713753964138e-01, 3.760111159247546e-01, 9.043554782179437e-01, 3.317940595212743e-02, 1.478494680325187e-01, 1.981697010664017e-01, 6.722702374574286e-01, 1.857782634085783e-01, 1.930398159770266e-01, 9.573452816191232e-01, 6.202600360777708e-01, 2.698836637044009e-01, 5.246373453963109e-01, 4.760797182674563e-01, 9.081024165069499e-01, 9.463249898054826e-01, };


#define UNCONSTRAINED


/* Optimal solution */

double xopt[30] = { -4.304466031806435e-01, -7.201806417165169e-02, -6.786213555453569e-02, 4.023354937954471e-02, -3.249713631608997e-02, -9.125265136741545e-02, 5.949861868434818e-01, 8.355241272738967e-02, -3.049517343405683e-02, -2.548113142531914e-01, -2.788110176670635e-01, 2.520790272954931e-01, -3.882949147202219e-01, -2.581425137716937e-01, 1.818619510979675e-01, -3.390371369086411e-01, 4.682699142708369e-02, 1.123552300635139e-01, 1.045230188560644e-01, 1.241368169009316e-02, 7.702132076606383e-02, -2.341025311593592e-01, 2.304731698550206e-01, -1.084051941306767e-01, 2.101624251568707e-01, -2.013175243708483e-01, -2.904508666464942e-01, 1.332174253232843e-03, -8.713570849198078e-01, 5.672720009966353e-01, };
double uopt[18] = { 1.583724322763598e-01, -7.920164393848017e-03, -1.291138294111938e+00, 3.248543970398676e-02, -1.207249274669039e+00, 7.854116472052245e-01, -5.569050896604080e-01, 5.504532638634951e-02, -1.471513237354818e-01, 1.081876797092414e-01, -4.820650980737928e-01, -9.546506827790560e-01, -8.444839256298395e-01, -1.302404865822222e-01, -5.222440763372285e-01, -1.422021251538825e-01, -5.596490305378665e-01, -2.085016302365275e+00, };