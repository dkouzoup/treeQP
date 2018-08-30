
/* Dimensions */

int Nn = 12;
int ns[12] = { 1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 0, };
int nx[12] = { 2, 2, 2, 2, 5, 2, 3, 2, 2, 2, 2, 1, };
int nu[12] = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, };

/* Data */

double A[58] = { 2.874272674825427e-01, 5.016591067466902e-01, 7.615461856696468e-01, 7.624080487887140e-01, 9.064432665100022e-01, 3.926845759741115e-01, 2.485523384487209e-02, 6.714367965724598e-01, 1.857782634085783e-01, 1.930398159770266e-01, 3.416441046451417e-01, 9.328978958180805e-01, 6.423152345852985e-01, 2.212657301271046e-01, 8.370564455318806e-01, 9.710752314425135e-01, 8.463728876931734e-01, 5.059994558905483e-01, 2.788756111452306e-01, 7.466172218282013e-01, 2.369303841128702e-01, 9.573452816191232e-01, 3.181047261502632e-01, 1.338109853561260e-01, 6.714628894780307e-01, 5.709910754624057e-01, 1.697670660264885e-01, 1.476557771517368e-01, 4.760797182674563e-01, 9.081024165069499e-01, 5.521750267158347e-01, 3.293989274987663e-02, 4.242267099033061e-01, 7.288638680513739e-01, 4.983535825235517e-01, 8.089902671990448e-01, 3.565089334810461e-01, 7.324343448980741e-02, 1.667471294023248e-02, 8.009208820578482e-01, 1.425093249237905e-01, 4.784744729028632e-01, 2.568353541796335e-01, 3.690916888248605e-01, 7.349559254300144e-01, 9.541027864712136e-01, 5.428131131035512e-01, 5.401058323593658e-01, 9.636122009513548e-01, 4.888997861606039e-01, 2.203101005686330e-01, 2.262086408412859e-01, 6.240005543123440e-01, 3.279415969732488e-01, 8.029653159582369e-01, 9.994778586358921e-01, 4.707632458669568e-01, 8.568963277821933e-01, };
double B[37] = { 5.760559014965571e-01, 7.476628376417164e-01, 8.371706353770545e-01, 9.714996383112634e-01, 3.906675366175962e-01, 2.732167079999634e-01, 6.202600360777708e-01, 6.002621455076278e-01, 1.726045016476686e-01, 9.034673814322458e-02, 2.552622026435888e-01, 5.386292643555612e-02, 8.050632285589021e-01, 5.909914552748493e-01, 9.101878307281412e-01, 1.937659361666103e-01, 4.323677915343976e-01, 7.491597290680092e-01, 3.918448664758301e-02, 6.617649133654143e-01, 1.696088134543144e-01, 2.787840204390187e-01, 1.982217945138389e-01, 3.111096396040020e-01, 7.123457192458371e-02, 1.819804671690215e-01, 9.298892687067795e-02, 5.367878045128258e-01, 7.621097092111474e-01, 3.475671504487593e-01, 4.612317593914992e-01, 9.809781609321463e-01, 1.270369421948571e-01, 2.322401459617927e-01, 2.363246665822505e-02, 4.339047156695319e-02, 6.916251452013056e-01, };
double b[25] = { 6.455345059813216e-01, 1.232195183245062e-01, 5.693288543524799e-02, 4.503238127334530e-01, 1.519470798468442e-01, 3.971088427434517e-01, 8.585705312592099e-01, 9.110670533977940e-01, 6.996337672257625e-01, 7.251823550233875e-01, 2.298860788574965e-01, 4.513748547034482e-01, 3.826462295599585e-01, 9.463249898054826e-01, 7.636733236637612e-01, 5.588205505095603e-01, 1.950715332842609e-01, 3.268396483499757e-01, 4.634892477622432e-01, 9.332512027769413e-03, 6.393237621993564e-01, 9.173360408668452e-01, 6.074326104018551e-01, 1.108093212871500e-01, 9.789854666750385e-01, };
double Q[71] = { 2.714533643763321e+00, 8.771879063624219e-01, 8.771879063624219e-01, 4.331971618247370e+00, 2.486094870756784e+00, 2.197044987633690e-01, 2.197044987633690e-01, 6.870551842606805e+00, 7.851615810862808e+00, 7.030352835461506e-01, 7.030352835461506e-01, 4.388517410325207e+00, 6.520992047080701e+00, 2.830777124693161e-01, 2.830777124693161e-01, 2.013040901207281e-01, 9.079626830067568e+00, 5.657845688278502e-01, 4.666016354326611e-01, 7.203233604813635e-01, 3.073180801915996e-01, 5.657845688278502e-01, 9.627883178897712e+00, 4.452943993007188e-01, 7.389579467978931e-01, 7.646034127697445e-01, 4.666016354326611e-01, 4.452943993007188e-01, 7.251520506832442e+00, 4.494453145323174e-01, 5.141208695021990e-01, 7.203233604813635e-01, 7.389579467978931e-01, 4.494453145323174e-01, 2.663445348038338e+00, 2.908718044259669e-01, 3.073180801915996e-01, 7.646034127697445e-01, 5.141208695021990e-01, 2.908718044259669e-01, 1.178115527970558e+00, 9.504408883648159e+00, 4.483184022493522e-01, 4.483184022493522e-01, 3.998552822703556e+00, 4.978688400790590e+00, 7.460959128314986e-01, 5.983933046055420e-01, 7.460959128314986e-01, 3.172767799721844e+00, 6.829526642581658e-01, 5.983933046055420e-01, 6.829526642581658e-01, 4.897881571275806e+00, 1.056958782185212e+01, 4.375356185933448e-01, 4.375356185933448e-01, 4.253788854449650e+00, 2.999728144788987e+00, 3.220803986677535e-01, 3.220803986677535e-01, 4.580046722515890e+00, 9.004000400201269e+00, 6.466871401799414e-01, 6.466871401799414e-01, 4.363816530978247e+00, 2.490919336011867e+00, 7.161047920692746e-01, 7.161047920692746e-01, 4.778435841954852e+00, 1.621072898436368e+00, };
double R[21] = { 1.911867663216918e-01, 1.820319002270999e+00, 8.138191087716664e-01, 2.152720099314896e+00, 2.007692826063867e+00, 1.477891020944497e+00, 5.254017546484466e-01, 5.254017546484466e-01, 1.205292020015306e+00, 2.629940283846057e+00, 6.575948677274588e-01, 6.575948677274588e-01, 2.065615269067572e+00, 2.414192251049496e+00, 4.959680875817933e-01, 4.959680875817933e-01, 1.807090315966166e+00, 1.688157925517183e+00, 3.678636445133088e-01, 3.678636445133088e-01, 1.227429749327169e+00, };
double S[31] = { 3.164195137325431e-01, 6.996169863970555e-01, 2.567845632701479e-01, 9.758649885159509e-03, 5.765436121487222e-02, 9.797652239759865e-01, 2.353667731508694e-01, 4.480197134640509e-01, 9.481087353960221e-01, 6.102892919250924e-02, 5.846413033551106e-01, 2.851080856586419e-01, 8.277321734482632e-01, 2.638340415267952e-01, 7.587662650802039e-01, 9.952159811297520e-01, 1.865714441413692e-01, 9.638701299717145e-01, 1.156258791812592e-01, 5.144829323020528e-02, 3.043489456365734e-01, 5.801918331427124e-01, 5.309644523382815e-01, 8.139769268209339e-01, 8.984441371801204e-01, 4.292385431148273e-01, 3.343294196222197e-01, 4.365547823722682e-01, 4.921318035648636e-02, 4.963190151939023e-02, 9.110017541305393e-02, };
double q[27] = { 6.252551801790405e-01, 5.430621753438516e-01, 5.322830710608887e-01, 2.793919652529646e-01, 2.848237268606177e-01, 5.949742988095963e-01, 5.693581832849320e-01, 6.140144229084699e-02, 1.909864406973978e-01, 4.425299622028837e-01, 3.934115063675764e-01, 8.265739790427650e-01, 6.768710934384189e-01, 7.811452685347645e-01, 1.957979810267320e-01, 9.012080926531435e-01, 5.405504251702443e-01, 4.319806108566827e-01, 5.966471044452463e-01, 9.019908087065621e-01, 5.940370314441212e-01, 2.410840551690221e-01, 1.789751526277321e-01, 6.333335803201215e-01, 9.561961521758778e-01, 1.240259166480432e-01, 6.852796844126873e-01, };
double r[13] = { 4.390372033876692e-01, 9.462301535199267e-01, 9.621610310112598e-01, 4.962888856398848e-01, 2.076030343799805e-01, 9.923589731799265e-01, 8.022615697642888e-01, 5.426669874363416e-01, 7.124148057895217e-01, 7.020664493596758e-01, 3.774551020708649e-01, 8.413691019721437e-01, 8.572127640906055e-01, };

/* Optimal solution */

double xopt[27] = { -4.879128701824276e-02, -1.446773293420756e-02, 3.638409844711457e-01, -2.453956173442475e-01, -6.361560566752189e-02, -8.710203403737338e-02, -5.622332737734705e-01, -1.668202516430646e-01, -2.119570883359163e-01, 1.353065823114930e-01, -6.945763008641781e-02, 4.865169146745285e-02, -6.628844927031234e-01, 1.035025867410404e-01, 1.948735315178457e-01, 1.131537476100326e-01, -7.351551652216262e-01, 3.471988686517575e-01, -1.407338826629225e-01, 6.036055117998396e-02, 2.164493745636943e-01, -1.435951918636398e-01, 3.707226606206071e-02, -7.588041949814528e-02, -3.003758083249903e-01, -7.391182429077858e-02, 7.742189088276252e-02, };
double uopt[13] = { -4.455326778312103e-01, -5.306568545650503e-01, -1.721678914674257e+00, -1.007618788159973e+00, -3.089930175287703e-01, -1.310221157431169e+00, -6.022675391278911e-01, -3.174186774367238e-01, -4.018959068986274e-01, -4.430103293747598e-01, -2.118160680007590e-01, -6.539035360276813e-01, -1.231939897582939e+00, };
