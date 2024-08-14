#ifndef __FASTGRNN_FC_PARAMS__
#define __FASTGRNN_FC_PARAMS__

#define FC_IN_DIM (32)
#define FC_OUT_DIM (6)

// clang-format off

const float FC_W[6][32] = {
   {-2.53916055e-01, -4.14207429e-01,  1.40552485e+00,  5.34862518e-01, -3.17264527e-01, -5.31883053e-02,  3.98890734e-01,  2.90338606e-01,
     6.80515826e-01, -3.55508029e-01, -5.59751391e-01,  5.78341722e-01, -8.20680797e-01,  9.30706918e-01,  4.21637207e-01,  7.32617259e-01,
    -3.94288659e-01, -1.95914879e-01,  7.50968277e-01,  5.38720787e-01, -3.33731264e-01, -6.67075455e-01,  1.75678775e-01, -2.25446492e-01,
     2.25261867e-01,  4.36965793e-01,  5.57585418e-01,  5.16550481e-01,  2.57467389e-01,  3.08370799e-01,  4.21284676e-01,  3.07654530e-01},
   { 5.98745525e-01, -8.20106268e-01, -2.15610933e+00, -1.68851233e+00,  1.66205847e+00, -1.37247443e-02,  7.54817665e-01,  4.46881026e-01,
    -9.86541271e-01, -4.41901445e-01, -7.93072164e-01, -1.32182074e+00,  6.61624074e-01, -1.15357184e+00, -8.69253099e-01, -3.70422035e-01,
    -5.03577888e-01,  8.55582297e-01,  4.46988344e-01, -6.10901415e-01, -3.82465339e+00,  9.72647965e-01, -3.41207266e-01,  7.67652452e-01,
     3.62791121e-01,  5.06920457e-01, -2.25858879e+00,  8.12066555e-01,  1.14491570e+00, -1.66428900e+00, -1.48063636e+00, -2.11310339e+00},
   { 6.06734395e-01, -6.81808531e-01, -1.55162299e+00, -2.57645130e+00,  1.64318538e+00,  7.42380917e-02,  7.69993842e-01,  1.99278474e-01,
     9.87333715e-01, -3.26251179e-01, -5.92574596e-01,  1.05573583e+00,  7.53541768e-01,  1.15558493e+00,  1.62241197e+00, -5.65161586e-01,
    -4.43506867e-01, -4.88646686e-01, -1.97865093e+00, -1.41197515e+00,  1.53015602e+00, -6.55367434e-01, -4.36008632e-01,  7.48691380e-01,
    -2.70776570e-01, -1.12168491e+00, -1.39748287e+00,  7.88612187e-01, -1.78050375e+00, -1.38069642e+00,  4.35155600e-01, -2.29104900e+00},
   {-2.58117050e-01,  1.36686492e+00,  1.42472398e+00, -6.94910169e-01, -6.25756204e-01, -1.47488579e-01, -9.57569122e-01,  3.96342218e-01,
    -2.37148714e+00,  1.31667960e+00,  1.54547167e+00, -3.02483988e+00,  1.30087912e+00, -2.07968426e+00, -1.36123776e+00, -7.93602109e-01,
     1.10830927e+00, -2.66105920e-01,  3.74693811e-01, -1.74628878e+00, -4.64191437e+00,  1.79337418e+00,  5.77647626e-01, -5.82094193e-01,
    -1.23931015e+00, -1.37255859e+00, -3.09684062e+00,  4.69904572e-01, -1.25338426e-02,  7.35499382e-01, -8.95894170e-01, -2.68939853e+00},
   {-2.50098854e-01, -4.49667156e-01,  1.10673106e+00, -2.57743180e-01, -6.30379856e-01,  1.22398682e-01, -2.06497765e+00, -5.80233216e-01,
    -1.68003583e+00, -4.44268078e-01,  1.91766310e+00, -8.14692676e-01, -1.07045007e+00, -9.82154429e-01, -1.82696927e+00, -1.24555707e+00,
     9.81961727e-01, -3.45944196e-01,  7.44951248e-01, -3.27665716e-01,  2.11460185e+00, -1.26695824e+00,  1.50076166e-01, -3.11045796e-01,
     4.38387871e-01, -6.57831430e-01,  1.58444011e+00, -1.60731876e+00,  9.10903156e-01,  6.74542844e-01,  6.16465986e-01,  1.60372734e+00},
   {-2.56618470e-01,  1.89580667e+00, -1.75285804e+00,  1.50766945e+00, -5.58854163e-01, -6.80000409e-02, -1.69641924e+00, -1.30454779e+00,
    -8.66776764e-01,  1.22260666e+00, -5.87902367e-01, -4.13719922e-01,  1.07466328e+00, -2.27542639e+00, -2.34263396e+00,  7.10362017e-01,
    -4.60013121e-01,  8.37428927e-01, -1.77717328e+00,  1.16169500e+00, -1.52774608e+00,  1.46768403e+00,  4.40182894e-01, -5.08074939e-01,
    -4.75651324e-01,  8.82904887e-01, -1.79775691e+00, -1.64556587e+00, -8.50898325e-01,  7.06632972e-01, -8.81726623e-01,  1.00182605e+00},
};

const float FC_B[6] = 
   { 1.30214942e+00, -5.15856028e-01, -1.44853425e+00,  1.10192738e-01, -1.26923013e+00, -8.90121698e-01};

// clang-format on


#endif // __FASTGRNN_FC_PARAMS__

