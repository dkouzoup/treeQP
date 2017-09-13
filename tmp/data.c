#define maxit 100

/* Dimensions */

int Nh = 5;
int Nr = 3;
int md = 2;
int NX = 10;
int NU = 4;
double termTolerance = 1.000000e-06;
double typeREG = TREEQP_ALWAYS_LEVENBERG_MARQUARDT;
double tolREG = 1.000000e-06;
double valueREG = 1.000000e-06;
double gammaLS = 1.000000e-01;
double betaLS = 8.000000e-01;
int maxIterLS = 100;
double A[] = { 8.081794328613426e-01, 9.348699860111614e-02, 1.601061785474544e-03, 1.079750320605844e-05, 3.880920744730239e-08, -7.351735647677646e-01, 3.485031463221359e-01, 1.255028355590039e-02, 1.283309063997676e-04, 6.174878023257022e-07, 9.348699860111613e-02, 8.097804946468172e-01, 9.349779610432220e-02, 1.601100594681991e-03, 1.079750320605844e-05, 3.485031463221357e-01, -7.226232812118644e-01, 3.486314772285357e-01, 1.255090104370271e-02, 1.283309063997675e-04, 1.601061785474544e-03, 9.349779610432223e-02, 8.097805334560246e-01, 9.349779610432218e-02, 1.601061785474544e-03, 1.255028355590039e-02, 3.486314772285357e-01, -7.226226637240619e-01, 3.486314772285355e-01, 1.255028355590038e-02, 1.079750320605844e-05, 1.601100594681992e-03, 9.349779610432221e-02, 8.097804946468172e-01, 9.348699860111610e-02, 1.283309063997675e-04, 1.255090104370271e-02, 3.486314772285357e-01, -7.226232812118641e-01, 3.485031463221358e-01, 3.880920744730257e-08, 1.079750320605846e-05, 1.601061785474544e-03, 9.348699860111613e-02, 8.081794328613426e-01, 6.174878023257045e-07, 1.283309063997676e-04, 1.255028355590039e-02, 3.485031463221358e-01, -7.351735647677647e-01, 4.674889809545787e-01, 1.601100594945129e-02, 1.619638469937678e-04, 7.761894117364992e-07, 2.164829414683352e-09, 8.081794328613426e-01, 9.348699860111614e-02, 1.601061785474544e-03, 1.079750320605847e-05, 3.880920744730264e-08, 1.601100594945128e-02, 4.676509448015723e-01, 1.601178213886301e-02, 1.619660118231824e-04, 7.761894117364945e-07, 9.348699860111610e-02, 8.097804946468171e-01, 9.349779610432217e-02, 1.601100594681991e-03, 1.079750320605843e-05, 1.619638469937677e-04, 1.601178213886302e-02, 4.676509469664016e-01, 1.601178213886302e-02, 1.619638469937678e-04, 1.601061785474544e-03, 9.349779610432220e-02, 8.097805334560246e-01, 9.349779610432220e-02, 1.601061785474544e-03, 7.761894117364964e-07, 1.619660118231825e-04, 1.601178213886303e-02, 4.676509448015724e-01, 1.601100594945128e-02, 1.079750320605845e-05, 1.601100594681992e-03, 9.349779610432224e-02, 8.097804946468171e-01, 9.348699860111614e-02, 2.164829414683565e-09, 7.761894117364951e-07, 1.619638469937680e-04, 1.601100594945127e-02, 4.674889809545786e-01, 3.880920744730311e-08, 1.079750320605846e-05, 1.601061785474544e-03, 9.348699860111612e-02, 8.081794328613426e-01, 7.627210475938566e-01, 1.148825465938981e-01, 2.476544740668230e-03, 2.093807494018838e-05, 9.422244178312972e-08, -8.994147677423243e-01, 4.202389763578713e-01, 1.931209917588990e-02, 2.482501671044522e-04, 1.497052645657269e-06, 1.148825465938981e-01, 7.651975923345248e-01, 1.149034846688382e-01, 2.476638963110013e-03, 2.093807494018835e-05, 4.202389763578714e-01, -8.801026685664344e-01, 4.204872265249757e-01, 1.931359622853555e-02, 2.482501671044521e-04, 2.476544740668231e-03, 1.149034846688382e-01, 7.651976865569666e-01, 1.149034846688383e-01, 2.476544740668228e-03, 1.931209917588990e-02, 4.204872265249758e-01, -8.801011715137886e-01, 4.204872265249759e-01, 1.931209917588990e-02, 2.093807494018837e-05, 2.476638963110010e-03, 1.149034846688383e-01, 7.651975923345248e-01, 1.148825465938981e-01, 2.482501671044519e-04, 1.931359622853554e-02, 4.204872265249758e-01, -8.801026685664344e-01, 4.202389763578713e-01, 9.422244178312826e-08, 2.093807494018827e-05, 2.476544740668230e-03, 1.148825465938981e-01, 7.627210475938566e-01, 1.497052645657253e-06, 2.482501671044517e-04, 1.931209917588990e-02, 4.202389763578714e-01, -8.994147677423243e-01, 4.596139397276020e-01, 1.981311171287963e-02, 2.512600560286713e-04, 1.507575067593731e-06, 5.261210968232149e-09, 7.627210475938566e-01, 1.148825465938981e-01, 2.476544740668228e-03, 2.093807494018840e-05, 9.422244178312931e-08, 1.981311171287965e-02, 4.598651997836304e-01, 1.981461928794722e-02, 2.512653172396394e-04, 1.507575067593729e-06, 1.148825465938981e-01, 7.651975923345249e-01, 1.149034846688383e-01, 2.476638963110010e-03, 2.093807494018840e-05, 2.512600560286713e-04, 1.981461928794723e-02, 4.598652050448416e-01, 1.981461928794724e-02, 2.512600560286714e-04, 2.476544740668228e-03, 1.149034846688383e-01, 7.651976865569666e-01, 1.149034846688383e-01, 2.476544740668231e-03, 1.507575067593723e-06, 2.512653172396390e-04, 1.981461928794724e-02, 4.598651997836307e-01, 1.981311171287965e-02, 2.093807494018831e-05, 2.476638963110010e-03, 1.149034846688383e-01, 7.651975923345247e-01, 1.148825465938981e-01, 5.261210968231867e-09, 1.507575067593721e-06, 2.512600560286711e-04, 1.981311171287965e-02, 4.596139397276018e-01, 9.422244178312738e-08, 2.093807494018833e-05, 2.476544740668229e-03, 1.148825465938981e-01, 7.627210475938566e-01, 8.081794328613426e-01, 9.348699860111614e-02, 1.601061785474544e-03, 1.079750320605844e-05, 3.880920744730239e-08, -7.351735647677646e-01, 3.485031463221359e-01, 1.255028355590039e-02, 1.283309063997676e-04, 6.174878023257022e-07, 9.348699860111613e-02, 8.097804946468172e-01, 9.349779610432220e-02, 1.601100594681991e-03, 1.079750320605844e-05, 3.485031463221357e-01, -7.226232812118644e-01, 3.486314772285357e-01, 1.255090104370271e-02, 1.283309063997675e-04, 1.601061785474544e-03, 9.349779610432223e-02, 8.097805334560246e-01, 9.349779610432218e-02, 1.601061785474544e-03, 1.255028355590039e-02, 3.486314772285357e-01, -7.226226637240619e-01, 3.486314772285355e-01, 1.255028355590038e-02, 1.079750320605844e-05, 1.601100594681992e-03, 9.349779610432221e-02, 8.097804946468172e-01, 9.348699860111610e-02, 1.283309063997675e-04, 1.255090104370271e-02, 3.486314772285357e-01, -7.226232812118641e-01, 3.485031463221358e-01, 3.880920744730257e-08, 1.079750320605846e-05, 1.601061785474544e-03, 9.348699860111613e-02, 8.081794328613426e-01, 6.174878023257045e-07, 1.283309063997676e-04, 1.255028355590039e-02, 3.485031463221358e-01, -7.351735647677647e-01, 4.674889809545787e-01, 1.601100594945129e-02, 1.619638469937678e-04, 7.761894117364992e-07, 2.164829414683352e-09, 8.081794328613426e-01, 9.348699860111614e-02, 1.601061785474544e-03, 1.079750320605847e-05, 3.880920744730264e-08, 1.601100594945128e-02, 4.676509448015723e-01, 1.601178213886301e-02, 1.619660118231824e-04, 7.761894117364945e-07, 9.348699860111610e-02, 8.097804946468171e-01, 9.349779610432217e-02, 1.601100594681991e-03, 1.079750320605843e-05, 1.619638469937677e-04, 1.601178213886302e-02, 4.676509469664016e-01, 1.601178213886302e-02, 1.619638469937678e-04, 1.601061785474544e-03, 9.349779610432220e-02, 8.097805334560246e-01, 9.349779610432220e-02, 1.601061785474544e-03, 7.761894117364964e-07, 1.619660118231825e-04, 1.601178213886303e-02, 4.676509448015724e-01, 1.601100594945128e-02, 1.079750320605845e-05, 1.601100594681992e-03, 9.349779610432224e-02, 8.097804946468171e-01, 9.348699860111614e-02, 2.164829414683565e-09, 7.761894117364951e-07, 1.619638469937680e-04, 1.601100594945127e-02, 4.674889809545786e-01, 3.880920744730311e-08, 1.079750320605846e-05, 1.601061785474544e-03, 9.348699860111612e-02, 8.081794328613426e-01, };
double B[] = { 1.209020879409956e-01, 2.028466958669576e-03, 1.359422773868418e-05, 4.872865097102586e-08, 1.085708309487734e-10, 4.674889809545786e-01, 1.601100594945128e-02, 1.619638469937678e-04, 7.761894117364996e-07, 2.164829414683368e-09, 2.028466958669576e-03, 1.209156821687343e-01, 2.028515687320549e-03, 1.359433630951510e-05, 4.872865097102594e-08, 1.601100594945128e-02, 4.676509448015724e-01, 1.601178213886302e-02, 1.619660118231825e-04, 7.761894117364946e-07, 1.359422773868418e-05, 2.028515687320548e-03, 1.209156822773052e-01, 2.028515687320547e-03, 1.359422773868417e-05, 1.619638469937677e-04, 1.601178213886302e-02, 4.676509469664017e-01, 1.601178213886301e-02, 1.619638469937677e-04, 4.872865097102647e-08, 1.359433630951513e-05, 2.028515687320548e-03, 1.209156821687343e-01, 2.028466958669577e-03, 7.761894117364998e-07, 1.619660118231825e-04, 1.601178213886302e-02, 4.676509448015722e-01, 1.601100594945128e-02, 1.198988285101330e-01, 2.518704614122532e-03, 2.112731201018829e-05, 9.475056607222013e-08, 2.640621445454916e-10, 4.596139397276019e-01, 1.981311171287964e-02, 2.512600560286713e-04, 1.507575067593732e-06, 5.261210968232208e-09, 2.518704614122535e-03, 1.199199558221432e-01, 2.518799364688606e-03, 2.112757607233283e-05, 9.475056607222000e-08, 1.981311171287965e-02, 4.598651997836306e-01, 1.981461928794724e-02, 2.512653172396396e-04, 1.507575067593732e-06, 2.112731201018829e-05, 2.518799364688603e-03, 1.199199560862053e-01, 2.518799364688607e-03, 2.112731201018826e-05, 2.512600560286712e-04, 1.981461928794723e-02, 4.598652050448416e-01, 1.981461928794724e-02, 2.512600560286715e-04, 9.475056607221885e-08, 2.112757607233274e-05, 2.518799364688607e-03, 1.199199558221432e-01, 2.518704614122533e-03, 1.507575067593718e-06, 2.512653172396393e-04, 1.981461928794724e-02, 4.598651997836308e-01, 1.981311171287965e-02, 1.209020879409956e-01, 2.028466958669576e-03, 1.359422773868418e-05, 4.872865097102586e-08, 1.085708309487734e-10, 4.674889809545786e-01, 1.601100594945128e-02, 1.619638469937678e-04, 7.761894117364996e-07, 2.164829414683368e-09, 2.028466958669576e-03, 1.209156821687343e-01, 2.028515687320549e-03, 1.359433630951510e-05, 4.872865097102594e-08, 1.601100594945128e-02, 4.676509448015724e-01, 1.601178213886302e-02, 1.619660118231825e-04, 7.761894117364946e-07, 1.359422773868418e-05, 2.028515687320548e-03, 1.209156822773052e-01, 2.028515687320547e-03, 1.359422773868417e-05, 1.619638469937677e-04, 1.601178213886302e-02, 4.676509469664017e-01, 1.601178213886301e-02, 1.619638469937677e-04, 4.872865097102647e-08, 1.359433630951513e-05, 2.028515687320548e-03, 1.209156821687343e-01, 2.028466958669577e-03, 7.761894117364998e-07, 1.619660118231825e-04, 1.601178213886302e-02, 4.676509448015722e-01, 1.601100594945128e-02, };
double b[] = { 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, };
double dR[] = { 1.000000000000000e+00, 1.000000000000000e+00, 1.000000000000000e+00, 1.000000000000000e+00, };
double dQ[] = { 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, };
double dP[] = { 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, 1.000000000000000e+01, };
double r[] = { -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, };
double q[] = { -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, };
double p[] = { -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, -0.000000000000000e+00, };
double umin[] = { -1.000000000000000e+00, -1.000000000000000e+00, -1.000000000000000e+00, -1.000000000000000e+00, };
double umax[] = { 1.000000000000000e+00, 1.000000000000000e+00, 1.000000000000000e+00, 1.000000000000000e+00, };
double xmin[] = { -2.000000000000000e+00, -2.000000000000000e+00, -2.000000000000000e+00, -2.000000000000000e+00, -2.000000000000000e+00, -3.000000000000000e+00, -3.000000000000000e+00, -3.000000000000000e+00, -3.000000000000000e+00, -3.000000000000000e+00, };
double xmax[] = { 2.000000000000000e+00, 2.000000000000000e+00, 2.000000000000000e+00, 2.000000000000000e+00, 2.000000000000000e+00, 3.000000000000000e+00, 3.000000000000000e+00, 3.000000000000000e+00, 3.000000000000000e+00, 3.000000000000000e+00, };