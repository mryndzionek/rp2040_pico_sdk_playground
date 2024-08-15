#ifndef __FASTGRNN_GRNN1_PARAMS__
#define __FASTGRNN_GRNN1_PARAMS__

#define GRNN1_HIDD_DIM0 (64)
#define GRNN1_HIDD_DIM1 (32)

// clang-format off

const float GRNN1_ZETA = (-3.179134e-01);
const float GRNN1_NU = (-1.398317e+01);

const float GRNN1_BIAS_UPDATE[32] = 
   { 2.51099020e-01, -3.40871364e-01,  9.50829220e+00,  2.43003264e-01,  1.46638203e+00,  1.01959610e+00,  9.20209789e+00,  2.93648541e-01,
     8.14089203e+00,  1.57828581e+00,  7.84744501e-01, -7.33320773e-01,  5.21018600e+00,  1.03871000e+00,  1.20444956e+01,  1.10256090e+01,
     1.00903206e+01,  1.13985558e+01,  1.29014673e+01,  1.05072355e+01,  1.53720129e+00,  8.95703793e+00,  7.60688186e-01,  1.34158001e+01,
     1.10579920e+01,  1.18589401e+01,  1.01206522e+01,  3.94920826e-01,  8.47046673e-02,  1.21128826e+01,  1.18760300e+01,  9.53963757e+00};

const float GRNN1_BIAS_GATE[32] = 
   { 4.40030396e-01,  8.11804056e-01,  5.25691032e+00,  1.46983698e-01,  2.67326951e+00,  2.04528379e+00,  3.00848269e+00, -2.87111402e-01,
     3.44794679e+00,  3.22095656e+00,  1.88237214e+00, -5.45198584e+00,  4.64822197e+00,  3.03606987e+00,  4.73004532e+00,  4.13890982e+00,
     3.80157185e+00,  3.29655313e+00,  4.86674118e+00,  5.02194548e+00,  3.60872221e+00,  2.54989171e+00,  2.16852355e+00,  4.74147654e+00,
     2.33195686e+00,  4.11882448e+00,  2.80898476e+00,  5.35035551e-01, -5.09803772e+00,  4.13727808e+00,  4.95474625e+00,  3.89879608e+00};

const float GRNN1_W1[16][64] = {
   {-2.15938956e-01, -1.48581453e-02,  5.21151662e-01,  4.85704571e-01,  1.51086003e-01,  1.81497131e-02, -3.14588279e-01,  1.14534751e-01,
    -4.52075124e-01,  1.33832330e-02, -3.65223050e-01, -2.67612678e-03, -8.79127800e-01, -4.13817108e-01,  2.86016846e+00, -3.46718758e-01,
    -9.35584724e-01,  1.92861545e+00, -5.21262875e-03,  7.23144352e-01,  9.07042205e-01,  1.42069900e+00, -6.70285165e-01,  1.63737044e-01,
    -3.46325517e-01,  5.10416865e-01,  1.26249063e+00,  4.16955024e-01, -4.09545228e-02, -4.06208783e-01, -4.21105862e-01,  1.00143380e-01,
    -1.33338615e-01,  2.91159242e-01,  1.71893501e+00,  1.92356333e-01,  3.66022319e-01, -6.73799932e-01, -2.95934319e-01, -9.79742289e-01,
     4.72916901e-01,  1.08417660e-01, -3.21267903e-01,  3.07744980e-01,  1.88226902e+00,  1.14529943e-02,  1.70634031e+00, -7.82661289e-02,
    -3.50946724e-01, -3.81859615e-02,  1.85463524e+00, -3.55612636e-01,  4.75749403e-01,  1.41583323e-01,  1.05110013e+00, -3.41453224e-01,
     5.83265834e-02, -1.68671310e-01,  2.38200235e+00,  6.22191839e-02,  9.39287543e-02, -5.92219904e-02,  1.22967517e+00,  1.15907562e+00},
   { 1.15097925e-01,  4.27504331e-02, -3.95804107e-01,  4.56903458e-01, -3.05125982e-01, -2.13654235e-01,  2.40612566e-01, -2.33915639e+00,
    -3.10721010e-01, -4.04146612e-01,  2.75035590e-01, -1.53245163e+00, -4.16880369e-01, -6.80198669e-01,  6.38993502e-01, -7.51276836e-02,
    -6.60233498e-01, -3.63678128e-01, -3.26131463e-01, -3.38859707e-01, -6.59104407e-01,  5.73170841e-01, -1.99122608e+00,  1.71034887e-01,
    -6.43294752e-02,  3.72665197e-01, -2.82554626e-01,  6.91979885e-01, -4.13793534e-01,  1.11658657e+00, -5.66573858e-01,  4.52389657e-01,
    -4.62352395e-01, -1.51509726e+00, -2.17232317e-01,  6.91757023e-01,  2.35164955e-01,  2.39769444e-01, -2.32357696e-01, -3.48806530e-01,
    -2.52740920e-01, -1.79069445e-01, -1.51040912e+00, -1.18523645e+00,  7.96738267e-02, -7.26324141e-01, -1.50743401e+00, -5.31516612e-01,
    -3.14679384e-01,  9.34107900e-02, -6.27995357e-02,  8.44893605e-02, -1.14168406e+00, -1.64926147e+00, -1.37450203e-01, -1.97193608e-01,
    -9.21660542e-01,  3.88608515e-01, -2.89822727e-01, -1.48267031e+00,  3.22568804e-01, -2.87017524e-01, -2.15517998e-01,  1.06628127e-01},
   { 1.86458807e-02,  4.99691397e-01, -7.55463123e-01,  5.70248783e-01,  5.77360451e-01,  1.98411480e-01, -2.40172476e-01, -6.05468988e-01,
    -4.05964494e-01,  1.47999883e+00, -8.93870950e-01, -4.17449266e-01, -8.25707614e-01,  7.71852732e-01,  5.15489817e-01, -1.32384515e+00,
    -6.34171605e-01,  7.40018249e-01,  1.71269298e-01,  1.20530119e-02, -2.00755745e-01,  8.58345509e-01, -1.48336387e+00, -1.02513576e+00,
     1.24782717e+00,  4.66340035e-01, -4.07189608e-01, -3.54974657e-01, -1.16007960e+00,  1.20002162e+00, -4.60994333e-01,  2.12466225e-01,
     3.96688372e-01, -7.23734915e-01, -1.01168379e-01,  2.04851180e-01,  5.47964394e-01, -2.86093727e-02, -1.11280918e+00,  2.85324901e-01,
     1.02535141e+00, -1.89751804e-01, -5.41868925e-01, -1.40117139e-01,  9.07056630e-02,  4.32663947e-01,  4.29939628e-01, -7.59379938e-02,
     5.92181124e-02,  6.16095603e-01,  1.23474561e-02,  3.66871320e-02, -5.25016665e-01, -5.61889887e-01, -1.56302467e-01,  9.77296114e-01,
     8.18207934e-02,  3.30805838e-01,  6.73300445e-01,  2.55435765e-01,  6.42787039e-01, -3.09234440e-01, -6.00899458e-01, -4.82927591e-01},
   {-3.21152449e-01, -6.72064662e-01,  3.67985785e-01,  4.61106628e-01,  9.03433621e-01,  6.30433083e-01,  4.46998179e-01,  5.93829691e-01,
     1.73849389e-01,  3.71118128e-01,  4.25893366e-01, -2.28419495e+00,  1.38787413e+00,  2.96237618e-01, -7.67525792e-01, -2.76687324e-01,
    -1.54851055e+00,  9.22123790e-01, -3.44226688e-01,  1.73969656e-01,  5.91138065e-01,  1.38340032e+00,  4.57426906e-01, -5.39524436e-01,
    -3.30549449e-01,  5.24671018e-01, -8.69398355e-01,  3.96975249e-01,  3.94011617e-01,  1.89431012e-02, -6.54630423e-01,  2.86961555e-01,
    -1.50215268e+00,  2.27602229e-01,  1.73261786e+00, -1.42365408e+00,  9.17208076e-01, -1.81284934e-01, -1.69902474e-01, -1.39020789e+00,
    -7.62230754e-01,  3.24640960e-01,  3.07031460e-02,  8.57953355e-02,  2.69782037e-01, -1.07704973e+00, -1.42885673e+00, -6.30805850e-01,
     2.52235293e-01,  2.63338536e-01, -9.02931929e-01,  1.28092086e+00, -1.78308988e+00, -1.68103993e+00,  6.00533247e-01, -1.32318407e-01,
    -1.11460477e-01,  5.66100121e-01, -1.06411427e-01, -9.78041217e-02,  1.27333537e-01,  3.91195357e-01, -1.12676632e+00,  1.03465152e+00},
   {-6.23233140e-01, -2.31869221e-01,  6.96677208e-01,  4.87268150e-01, -3.49295288e-01,  7.34272227e-02, -5.46908714e-02, -2.59081833e-02,
    -9.30999398e-01, -4.09042031e-01, -1.04323804e+00,  3.08232248e-01,  5.33150256e-01,  4.79001671e-01, -1.92446023e-01,  3.75061408e-02,
    -5.67412198e-01,  7.88481534e-01,  3.89998585e-01, -6.93501294e-01, -4.57269400e-01,  6.66641891e-02,  3.06467682e-01, -3.69045466e-01,
    -2.46884257e-01,  2.01168761e-01, -1.06232417e+00, -4.34626818e-01, -7.28804410e-01, -9.04737830e-01,  1.31041431e+00,  1.24247938e-01,
     3.56190205e-01, -1.89720735e-01,  7.88612485e-01, -5.44698238e-01,  2.69702524e-01, -9.94990468e-01, -4.09131885e-01,  1.45333266e+00,
    -1.54571623e-01,  4.27092254e-01, -4.91019756e-01,  1.45443726e+00, -8.69112849e-01, -4.27211344e-01,  1.03068018e+00, -5.83715796e-01,
     5.93520880e-01,  4.80348840e-02, -2.27408075e+00,  2.43136868e-01, -4.09785323e-02, -1.96718618e-01, -2.34371454e-01, -3.80739003e-01,
     7.89789140e-01,  1.26589537e-01, -1.18282115e+00,  1.10815561e+00,  2.94239730e-01, -2.69528657e-01, -4.47766960e-01, -1.26704857e-01},
   { 4.23984259e-01, -3.79337609e-01, -6.38227761e-01, -8.03594291e-01, -6.27407968e-01, -6.90239761e-03, -2.76496768e-01,  1.82453990e-01,
    -1.70285001e-01, -9.61310044e-03,  4.28980142e-01, -1.22128630e+00,  1.94347918e-01,  4.30181682e-01,  1.44076154e-01,  6.58320248e-01,
    -9.83102858e-01,  2.46601391e+00,  5.16477764e-01,  4.12254989e-01,  1.23656547e+00, -4.70813781e-01,  7.31083274e-01,  6.79273784e-01,
     8.31857622e-01,  2.89420605e-01, -1.02691844e-01, -8.07199776e-02,  8.52349877e-01, -1.80814040e+00,  1.88382223e-01, -6.16418481e-01,
     2.35114127e-01, -1.15216482e+00,  7.87524104e-01,  3.44469883e-02,  3.41949286e-03, -6.32651523e-02,  5.03430128e-01, -7.45652080e-01,
     3.42012793e-02, -7.14646280e-01, -2.87916735e-02, -1.96475089e-02,  7.89340556e-01, -1.40044436e-01, -4.61736053e-01, -1.04043186e+00,
     8.97387087e-01,  8.57726038e-01, -1.02745044e+00, -4.56782915e-02, -1.56169474e-01, -4.63464946e-01,  6.53249323e-01,  4.01017457e-01,
     6.31800741e-02,  4.12802100e-01,  2.12998241e-01, -7.43549287e-01, -1.61819190e-01,  5.22207856e-01,  7.61636078e-01, -2.04262450e-01},
   { 1.45026624e-01, -1.02190268e+00,  1.12928651e-01,  7.35273242e-01, -1.75426260e-01, -2.10848033e-01,  6.44975007e-01, -1.11273177e-01,
     2.42902562e-01,  8.21810484e-01,  5.04305102e-02, -6.35304332e-01,  2.20428571e-01,  6.09757960e-01,  1.74507856e+00, -4.78647426e-02,
    -8.50383997e-01, -1.60698926e+00,  9.03117478e-01, -6.43589497e-01,  9.04942513e-01,  7.68732369e-01, -1.74178708e+00,  8.00951421e-01,
     3.97611707e-02, -4.89111125e-01,  5.91821432e-01,  4.22160178e-01,  5.90751708e-01,  2.41125941e-01, -1.60903835e+00, -4.51801926e-01,
    -2.05804825e-01,  9.29671645e-01,  5.07945597e-01,  1.23384058e+00, -9.05496478e-02,  4.95260358e-01,  2.81553399e-02,  3.01368326e-01,
     7.43931890e-01,  5.53574026e-01, -3.65201414e-01, -4.22662273e-02, -6.53055966e-01,  1.69945776e-01,  4.48153866e-03, -6.85513020e-01,
    -3.82708877e-01,  6.55371010e-01, -3.52816999e-01,  1.01727568e-01, -7.22384036e-01, -1.36309612e+00, -2.99515873e-01, -7.08733678e-01,
    -6.84169531e-01,  8.83465528e-01, -3.41395289e-02,  3.80335987e-01, -8.37616026e-01,  2.47772500e-01,  1.02872574e+00,  6.38773292e-02},
   {-1.03494549e+00, -1.11506510e+00, -5.21542013e-01, -8.83271873e-01, -6.33591115e-01, -4.59430844e-01, -8.84825826e-01,  1.18104286e-01,
    -4.28476602e-01,  3.42384785e-01, -3.97549570e-01, -4.80739027e-02, -8.54205370e-01,  4.63296115e-01,  2.39771917e-01, -3.21560770e-01,
    -3.08757216e-01, -2.94930935e-01, -1.24949920e+00, -1.86084315e-01,  6.88897967e-01, -8.10335696e-01,  5.32090664e-01,  8.15548480e-01,
     5.14092922e-01,  4.41581815e-01,  1.80253232e+00,  9.84003395e-02,  5.47237694e-01, -7.47797966e-01, -1.50251770e+00, -6.04867697e-01,
    -8.15782547e-01,  7.84339488e-01, -4.06004846e-01, -9.00203347e-01, -7.20769644e-01, -3.83545458e-01, -1.99810594e-01, -1.25473881e+00,
    -1.14947236e+00, -3.85388762e-01,  2.92981118e-01,  1.31911099e+00, -1.11080217e+00, -9.21525896e-01, -4.79306042e-01, -9.40419614e-01,
    -5.35137177e-01, -8.23218048e-01,  2.59124041e-01, -4.83342081e-01, -3.38537782e-01, -2.17335269e-01, -1.02700901e+00,  1.09126055e+00,
    -1.23127031e+00, -8.14507067e-01, -2.26497993e-01, -5.76439619e-01, -5.76680481e-01, -6.34715557e-01,  1.39569354e+00,  1.22211933e+00},
   { 1.55765548e-01, -7.00113550e-02, -1.39043644e-01, -1.08411932e+00,  1.81263700e-01,  1.91327482e-01, -1.41387951e+00,  9.16244507e-01,
    -2.80374676e-01,  2.29062796e-01, -1.55849814e-01,  2.15028644e-01,  4.64158244e-02, -5.43525398e-01,  3.25752348e-01, -1.03312626e-01,
     6.66238368e-01,  3.66310388e-01, -2.32529625e-01,  1.27613330e+00, -9.11111832e-01,  1.03471637e+00,  3.18211883e-01,  8.56803358e-01,
    -1.43314272e-01, -7.77607322e-01, -3.99404138e-01, -7.42758870e-01,  1.16665637e+00,  4.76051718e-02, -1.24653721e+00, -6.28069580e-01,
     3.45384568e-01, -3.41408581e-01,  7.51751736e-02, -4.37752903e-01, -8.66770968e-02,  5.00277936e-01,  5.00375688e-01, -4.65846807e-01,
     3.08865696e-01, -9.53882635e-02,  2.09654942e-01,  1.28993079e-01, -1.23245430e+00, -1.04847527e-03,  1.12661672e+00, -1.13146946e-01,
    -4.27160382e-01,  5.58375835e-01, -3.90734285e-01,  3.05921018e-01,  8.07122111e-01,  7.77639687e-01, -2.84151793e-01, -9.88913238e-01,
     7.05801189e-01,  2.25248739e-01, -1.10215461e+00, -2.99349993e-01, -7.64513671e-01, -5.77044725e-01, -3.60842317e-01, -3.26708734e-01},
   {-2.81682611e-01,  1.52205110e+00,  1.26291525e+00,  8.27915296e-02,  6.38498187e-01,  1.30604103e-01,  3.08395270e-02,  5.92094243e-01,
    -6.11826420e-01, -8.65971208e-01, -3.18779320e-01, -1.80881113e-01, -1.89077735e+00, -3.66833806e-01,  9.50425148e-01,  7.35562742e-01,
     1.36815941e+00,  6.96922660e-01, -1.76537490e+00, -7.11919487e-01, -6.38801396e-01, -7.25246787e-01, -1.11504018e+00,  1.06028879e+00,
    -6.13523185e-01,  7.07603872e-01,  9.03413832e-01,  1.12181735e+00, -2.21473813e-01,  1.26149321e+00, -4.68573809e-01,  5.81165433e-01,
    -3.64230007e-01, -4.30053681e-01, -1.31733930e+00,  3.51430297e-01,  7.35662878e-01, -1.18555450e+00,  2.47166485e-01, -2.22523183e-01,
    -5.94245732e-01,  1.09875941e+00,  2.57831573e-01,  1.98772371e-01, -1.17017925e+00, -7.54647017e-01,  7.74549365e-01,  4.34431374e-01,
     1.21071601e+00, -3.36830288e-01,  4.38716918e-01, -1.98031485e+00,  8.55999768e-01,  1.26350009e+00, -1.49202216e+00, -4.99840975e-01,
    -8.79885077e-01, -4.36908782e-01, -5.22105157e-01, -9.61817980e-01,  1.25247014e+00, -2.04892159e-01,  4.48301062e-02, -9.99020636e-02},
   { 7.70420551e-01,  4.63311642e-01,  1.53326809e-01, -2.93520838e-01, -3.16760033e-01,  1.99438512e-01,  2.63323605e-01,  7.09836423e-01,
     9.84930575e-01, -8.21366966e-01,  9.24795926e-01, -3.39798510e-01, -2.44767278e-01, -3.74038756e-01,  3.64630371e-01,  7.40699470e-01,
     5.34089446e-01,  1.20827222e+00,  3.69758576e-01,  8.30300808e-01,  1.43014401e-01, -2.87786245e-01, -2.24597692e-01,  5.87738693e-01,
    -3.77352893e-01, -8.08358610e-01, -7.50914991e-01, -5.52625656e-02,  1.26673925e+00, -5.63726783e-01, -5.73665380e-01,  2.97751799e-02,
    -8.33805203e-01, -1.41555810e+00,  5.18005550e-01,  3.75707686e-01, -1.56327561e-01, -4.72090989e-02,  7.95741439e-01, -4.40957278e-01,
     1.21006913e-01,  2.77260393e-01,  7.76709437e-01, -5.68995357e-01, -2.25614443e-01, -8.13819230e-01,  4.99633290e-02, -7.13047564e-01,
     1.52758449e-01,  8.20190430e-01, -7.21357584e-01, -2.82647341e-01,  7.26547480e-01,  5.74408293e-01, -4.63111363e-02, -1.36238360e+00,
    -4.83895272e-01,  7.40088105e-01, -1.28014219e+00,  1.95742235e-01, -9.23404917e-02,  1.09787655e+00, -3.68926793e-01, -9.85703647e-01},
   { 7.03917325e-01,  9.72259119e-02, -1.43647864e-01, -1.16700995e+00,  2.75822312e-01, -4.19926971e-01, -8.30888152e-01,  1.86536813e+00,
    -8.66261601e-01,  3.41950774e-01, -1.60545081e-01, -9.44132805e-01,  1.44846022e+00,  1.40918056e-02, -1.01969659e+00,  9.58522856e-01,
    -3.97899598e-01, -4.76207465e-01, -4.49676931e-01,  1.05679408e-01,  5.68118751e-01,  6.50502801e-01, -8.12042892e-01, -2.81805873e-01,
     4.02586102e-01,  2.92217761e-01, -8.63021553e-01, -3.73947650e-01,  3.55547041e-01,  6.53939426e-01,  5.76037049e-01, -3.20660204e-01,
     1.91269130e-01, -1.13016635e-01, -5.66800833e-01,  4.32582200e-01,  1.94200635e-01,  4.62327629e-01,  1.11972570e+00,  8.59184042e-02,
     1.42187446e-01, -1.87110245e-01,  1.40608579e-01, -6.45276129e-01,  7.87176251e-01,  5.33372164e-01,  1.33016989e-01,  7.52121747e-01,
     2.78283924e-01,  6.33461058e-01,  2.77776539e-01, -1.97220996e-01,  6.05219960e-01,  1.52420551e-01, -7.14090168e-01,  2.54894108e-01,
     4.60914820e-01,  2.41562605e-01,  1.59079170e+00, -1.68966222e+00,  8.10617805e-02, -1.36373913e+00,  1.54205009e-01, -7.06041515e-01},
   { 7.60942161e-01, -3.54237735e-01,  3.97983819e-01, -5.99115252e-01,  6.94834054e-01, -7.35565126e-01,  4.93102491e-01, -2.86064625e-01,
    -6.61065996e-01,  1.06470478e+00, -1.75723284e-01,  8.06111276e-01,  9.82488394e-02,  4.03818637e-01,  9.72675264e-01,  6.75265417e-02,
     2.61420965e-01,  6.27917528e-01,  1.08835530e+00,  6.02547348e-01,  9.83998597e-01, -5.10683239e-01,  3.12297910e-01,  5.41795433e-01,
     2.35435814e-01,  7.62206435e-01, -8.12735707e-02, -2.03723982e-01, -5.76752603e-01,  7.96029568e-01, -6.52809560e-01,  8.15424085e-01,
    -1.05450058e+00,  9.68557447e-02,  9.21251357e-01,  8.90787542e-01, -7.29197145e-01,  1.34051189e-01,  2.60293126e-01,  3.07836294e-01,
    -1.06765962e+00,  4.69991207e-01, -3.12250614e-01,  1.24163306e+00,  1.07650816e+00, -1.39786494e+00,  1.00254595e+00, -4.82864499e-01,
     2.42829800e-01,  6.49684444e-02,  2.36839801e-01,  1.03365791e+00,  5.81377506e-01,  3.54724020e-01,  2.60386944e-01,  4.26221460e-01,
    -4.17156279e-01,  7.62163579e-01,  6.86102092e-01,  1.47289872e-01, -4.75789994e-01, -6.20818436e-01,  2.21147403e-01, -9.40078720e-02},
   { 7.48887181e-01, -1.81538776e-01,  1.37368217e-01,  7.31048763e-01,  9.24617529e-01,  3.35695773e-01, -3.65419954e-01, -8.20416927e-01,
     3.22038531e-01,  1.09306288e+00,  2.06362009e-01,  2.99024254e-01, -8.27432811e-01,  1.29376557e-02,  3.87683451e-01, -1.00174177e+00,
     6.07837200e-01,  7.57234618e-02, -7.23469913e-01, -5.30454874e-01,  9.42887068e-01,  4.76985216e-01,  6.03005588e-01, -3.06081504e-01,
     5.42597547e-02,  6.81411147e-01, -1.74782979e+00, -3.04113656e-01,  2.20837131e-01, -4.05428559e-01,  3.90885994e-02,  2.44499996e-01,
     8.34156215e-01,  3.57174963e-01, -1.83056414e+00, -8.85298729e-01,  1.75369814e-01, -1.81976050e-01, -7.00390100e-01,  1.77289057e+00,
    -4.80476558e-01,  1.14951409e-01, -1.05351031e+00, -1.13890398e+00,  1.00612605e+00,  5.73978543e-01, -1.86623067e-01, -7.52941608e-01,
    -2.50426292e-01,  1.19407368e+00, -1.63296151e+00, -7.24823594e-01,  1.05729468e-01,  1.90583885e-01, -1.95709455e+00,  2.81187087e-01,
     1.40875299e-02, -1.55036390e-01,  5.41483127e-02,  5.91836512e-01, -7.78481126e-01,  2.55742781e-02, -1.17638278e+00,  8.10727656e-01},
   {-6.90851033e-01,  6.31252527e-01, -6.85918748e-01, -1.23035675e-03, -9.41676795e-02, -2.88257986e-01, -2.50204593e-01, -8.04233611e-01,
    -6.15300536e-01, -1.18978310e+00, -6.00558698e-01,  7.02534199e-01, -2.51157999e-01, -9.21278377e-04, -3.35136503e-01, -6.29092216e-01,
     2.20332026e+00, -1.37417078e+00, -4.88874495e-01,  1.18049920e-01,  7.11969256e-01, -9.22991216e-01, -7.68062651e-01,  7.52716959e-01,
    -6.49696290e-01,  1.12938739e-01, -7.21382380e-01, -6.51750937e-02, -7.51733005e-01,  2.11414382e-01, -7.88740039e-01,  4.86423612e-01,
    -7.56192327e-01,  1.03097653e+00,  1.12751400e+00,  1.07155897e-01,  3.38857435e-02, -5.60796224e-02,  2.64630854e-01, -1.28304586e-01,
     5.19145787e-01, -4.99087870e-01, -1.01745233e-01,  3.15645784e-01,  1.58863413e+00, -4.87550288e-01,  1.63331956e-01, -1.45797944e+00,
    -6.86649084e-02, -1.00167751e+00, -2.02301288e+00,  1.22244120e+00,  7.36260176e-01,  1.11992002e+00,  1.26154065e+00, -5.69716454e-01,
     2.40374967e-01, -5.95199093e-02,  8.64489749e-02, -3.95163745e-01,  1.13762593e+00, -6.44062519e-01,  4.17668939e-01, -9.71931875e-01},
   { 1.08481550e+00,  1.59840417e+00, -1.69066355e-01, -8.34652662e-01, -2.93553710e-01,  9.70224202e-01, -2.44239464e-01,  1.62797081e+00,
     1.16117287e+00,  5.43011963e-01,  1.48080909e+00,  7.02901363e-01, -4.34181482e-01, -5.03708534e-02, -1.13381970e+00,  1.26664098e-02,
     1.39108157e+00, -3.58546853e-01,  1.30015790e+00, -7.87686288e-01, -1.89589605e-01, -7.96224058e-01, -6.84497416e-01, -1.32839155e+00,
    -4.72007208e-02,  3.68633240e-01, -2.65977114e-01, -1.07703733e+00, -5.19592583e-01,  1.15456700e+00, -6.15379930e-01, -1.85631856e-01,
    -1.61464715e+00,  3.75159502e-01, -8.69311512e-01,  1.02741909e+00,  5.25843382e-01, -5.30520439e-01, -6.37447476e-01, -5.04711926e-01,
    -3.45392972e-01, -7.49937654e-01,  1.36017454e+00, -4.47362989e-01, -4.22246784e-01, -1.23166645e+00, -6.50428474e-01, -4.52125549e-01,
    -1.48718226e+00,  8.83186758e-01,  5.11510432e-01, -9.78159845e-01,  7.36986935e-01,  8.85569394e-01, -4.71441925e-01,  3.61593992e-01,
    -4.17576730e-01,  9.64010894e-01,  5.60711145e-01,  1.42364070e-01,  1.15577495e+00,  1.09516335e+00, -6.83089256e-01,  9.04214382e-01},
};

const float GRNN1_W2[32][16] = {
   {-1.51790053e-01,  3.50687414e-01, -1.24848716e-01, -5.89972019e-01, -5.09116575e-02, -5.65252841e-01,  1.81825832e-02,  1.28679231e-01,
    -9.04585198e-02,  6.38498306e-01, -4.40784078e-03,  2.11366937e-01,  3.97516221e-01,  1.26309441e-02,  6.50737405e-01, -1.16766684e-01},
   {-4.01669174e-01,  4.82028723e-01, -8.79478037e-01,  1.96172014e-01, -4.62012678e-01,  4.69531894e-01, -2.69686401e-01, -2.42626384e-01,
     5.91915727e-01,  5.27116098e-02,  1.53216636e+00,  1.18969120e-02,  2.43457165e-02, -1.11983605e-01, -2.83059597e-01, -6.37449324e-01},
   {-1.33431244e+00,  7.05337346e-01, -6.22870147e-01, -5.20853400e-01, -1.03561915e-01,  1.76193655e-01,  9.15652335e-01, -4.12100405e-01,
    -8.94836009e-01, -3.24532151e-01,  6.23504102e-01, -1.21974897e+00, -3.21814150e-01,  7.05377817e-01,  3.92305791e-01,  1.22387207e+00},
   { 2.48579979e-02, -3.34092170e-01,  9.39592868e-02, -9.60769057e-02,  2.53272146e-01,  1.65263787e-01,  8.08599964e-02,  3.88945282e-01,
    -5.11878669e-01, -1.56049982e-01, -1.30482242e-01,  1.29265375e-02, -2.58487433e-01, -8.63595977e-02, -2.10328668e-01,  5.20041406e-01},
   { 3.94444853e-01,  6.53480351e-01,  1.92174658e-01, -4.62123156e-01, -3.75772357e-01,  7.41038263e-01,  8.97330344e-01, -1.19915865e-01,
     1.67826965e-01,  1.72129080e-01,  5.27161777e-01,  2.59349823e-01, -5.91476262e-01, -8.63795638e-01, -1.82679743e-01, -3.49572569e-01},
   { 3.68248522e-01,  6.30790830e-01,  1.19592294e-01,  1.32568762e-01, -2.02559382e-01,  8.32757056e-01, -5.61193645e-01,  1.52318418e-01,
    -3.83794427e-01, -5.95299363e-01,  9.53230411e-02,  2.62857676e-01,  4.46179271e-01, -4.95240241e-01,  7.25001320e-02,  1.17935801e+00},
   { 2.81097025e-01, -3.91556293e-01, -3.97485167e-01, -1.10510454e-01,  4.88075823e-01,  7.56271601e-01, -2.61162937e-01, -1.98798859e+00,
     9.24083963e-02,  2.09259227e-01,  1.71153986e+00, -6.95386827e-01,  1.13949299e+00,  1.14599621e+00,  1.27469671e+00,  5.47873303e-02},
   {-8.92446041e-02,  3.66326541e-01,  1.87883466e-01,  2.03146309e-01, -3.52594048e-01, -1.67265564e-01, -4.21122201e-02, -8.42557400e-02,
     2.20141962e-01, -4.08269674e-01, -7.30393305e-02, -9.53175798e-02, -1.41115159e-01, -2.72958368e-01,  1.29695028e-01, -5.06798863e-01},
   { 1.76199675e+00,  1.20271325e+00,  6.66291356e-01,  1.25333178e+00,  4.56168383e-01, -1.25942826e-01, -1.10428107e+00, -9.08894241e-01,
    -1.90662503e-01,  8.07303429e-01,  3.69992435e-01, -6.35319725e-02, -1.21511984e+00,  1.19625616e+00, -2.42034048e-02,  3.36911887e-01},
   { 7.71626234e-01, -1.91043243e-01, -1.35798171e-01, -1.38196588e-01, -3.96391958e-01,  2.27421269e-01,  1.24636687e-01, -3.84522706e-01,
    -3.03658575e-01, -1.40818369e+00, -6.16622448e-01,  2.99129725e-01, -1.74779788e-01,  8.13755020e-02,  3.65643054e-01, -5.84618151e-01},
   { 5.02866447e-01, -3.85011971e-01, -4.32823479e-01,  5.27035475e-01,  2.91669995e-01, -8.54146242e-01,  1.22414559e-01, -1.26600480e+00,
     2.16003716e-01,  3.36945564e-01,  6.70380294e-01, -7.79286399e-02, -1.08584315e-01,  7.18325600e-02,  1.48950949e-01,  9.59379017e-01},
   { 3.98868844e-02, -1.30023509e-01, -6.01601064e-01,  1.27549410e+00, -6.69467032e-01,  2.49482408e-01, -2.70630777e-01,  8.95849466e-01,
     5.60034513e-02,  8.51751029e-01,  4.04896230e-01,  1.42753229e-01, -5.71107641e-02, -5.50633788e-01,  4.20115560e-01,  7.97770381e-01},
   {-1.44591987e-01, -8.97611007e-02,  7.55630732e-01, -1.38767517e+00,  3.73807937e-01, -1.86054632e-01,  1.86596796e-01, -5.50146341e-01,
    -3.53181988e-01, -9.48892295e-01, -6.44615114e-01, -3.03419828e-02,  4.32425022e-01,  5.31531930e-01,  5.45692205e-01, -5.32858551e-01},
   { 6.11864030e-01, -2.07442686e-01, -2.15652332e-01,  2.38170877e-01, -9.60344732e-01,  6.11757219e-01,  6.53615296e-01, -9.69449103e-01,
     6.15072131e-01, -8.07221770e-01,  1.66414201e+00, -2.43714750e-02,  5.68134606e-01, -1.14820868e-01,  1.36765361e-01,  1.74973607e+00},
   { 4.63953733e-01,  5.78268528e-01,  3.73691261e-01,  2.02000841e-01,  4.79113191e-01,  5.67574918e-01,  4.06043917e-01, -1.26897216e+00,
    -2.55298942e-01,  1.59120464e+00,  1.61160851e+00, -7.61433661e-01,  1.85498869e+00, -5.88192403e-01,  1.61189747e+00, -4.79263574e-01},
   {-9.08493221e-01, -4.90731925e-01, -1.27247244e-01,  3.50622416e-01,  1.50422466e+00,  5.31573296e-01, -5.67327380e-01, -2.79993510e+00,
     2.37122089e-01,  4.14858907e-01,  1.73876929e+00, -4.67791378e-01,  4.86666381e-01,  5.99926591e-01,  1.83402989e-02, -3.07375312e-01},
   {-1.10202658e+00,  9.82059956e-01, -5.86059093e-01, -6.51867509e-01,  7.04538882e-01,  8.28866243e-01,  3.89925897e-01, -7.17876852e-01,
     7.64308870e-01, -1.15492856e-02,  1.01727760e+00,  4.44237292e-01, -1.78614080e-01,  1.05084527e+00, -1.57609499e+00,  2.70622671e-01},
   { 1.46824527e+00, -1.88384950e-01, -1.26726925e+00,  1.35023677e+00,  1.47794211e+00,  1.04594624e+00,  7.16987431e-01, -1.14279318e+00,
     8.68850015e-03, -5.88104844e-01,  1.01908863e+00, -1.63861096e-01,  4.03877348e-01, -1.50261271e+00, -4.77277517e-01, -2.35335040e+00},
   {-3.54644917e-02, -5.43408036e-01, -1.58161651e-02,  1.03576744e+00,  4.74684909e-02,  1.55912387e+00,  5.78701422e-02, -2.15622997e+00,
    -4.91230607e-01, -1.23850417e+00,  1.36394846e+00,  5.35489500e-01,  7.30551362e-01, -9.86500978e-01, -5.02819300e-01,  2.32113671e+00},
   { 2.29570270e+00, -7.33644187e-01,  5.80561757e-01,  4.94404696e-04, -3.41376103e-03, -3.47940415e-01, -4.50698793e-01, -1.76039115e-01,
    -1.71201229e+00, -6.18514717e-01, -6.72533512e-01, -1.21612474e-01,  1.96342611e+00, -5.15452564e-01, -2.95064021e-02, -1.61379546e-01},
   {-6.84187353e-01,  1.35682151e-01, -1.08727701e-01, -8.54138285e-02, -8.46445188e-02, -4.25226599e-01,  1.12836687e-02, -4.15898301e-02,
    -6.15650356e-01, -2.51755327e-01,  3.28473508e-01, -1.08598256e+00, -4.20271784e-01,  4.05124784e-01,  2.11709246e-01,  7.46307850e-01},
   {-6.54098749e-01,  1.28300142e+00, -9.15474474e-01, -1.25598103e-01, -1.10503566e+00, -1.58013964e+00,  4.11679447e-01, -2.17742538e+00,
    -1.91513503e+00,  1.56710362e+00,  2.03329012e-01,  1.61668360e+00, -5.32919466e-02, -4.08201933e-01, -8.21678996e-01,  5.09490907e-01},
   { 7.76222646e-02, -9.57270980e-01,  3.69394034e-01, -3.72706056e-01, -4.97898199e-02,  8.74258459e-01, -1.98901042e-01, -5.46273112e-01,
     5.50682425e-01, -6.21685743e-01,  6.43717408e-01,  7.51360536e-01, -7.93233663e-02,  8.56838167e-01,  4.83021230e-01, -2.56272465e-01},
   {-9.13925171e-01, -1.77295506e-01, -2.69195676e-01, -1.56720176e-01, -6.91380918e-01,  1.46476483e+00, -5.32755077e-01,  3.24557908e-02,
     3.53892833e-01, -1.32833099e+00,  1.05079234e+00,  2.20120594e-01,  4.28879201e-01,  5.79955578e-01,  7.12559640e-01,  1.50263929e+00},
   { 1.94229281e+00, -1.31402421e+00, -1.54848707e+00, -2.06680059e+00,  1.73469216e-01, -1.02401219e-01, -1.29350412e+00, -6.39677763e-01,
     8.37038755e-01,  1.94948208e+00,  1.33009613e+00, -4.83156681e-01,  7.97050297e-01, -7.14405835e-01,  2.03123569e-01, -3.72106880e-01},
   { 2.04915524e+00,  1.67242837e+00,  1.47154105e+00,  1.28690195e+00, -5.81815124e-01, -1.38728464e+00,  5.28137922e-01, -8.94909739e-01,
    -4.28754449e-01,  1.73979664e+00, -2.04098254e-01, -2.07521558e-01,  1.88293561e-01,  1.03148103e+00, -2.36543894e+00, -1.10040927e+00},
   {-9.22593474e-01, -2.33829454e-01,  1.02426159e+00, -7.09673584e-01,  2.73501873e+00, -8.31235647e-01, -1.46070004e+00, -3.10564017e+00,
    -5.28151155e-01, -2.56987453e-01, -3.28340709e-01,  3.73447120e-01,  3.20865303e-01,  3.98188561e-01, -1.44368500e-01, -7.12378979e-01},
   { 1.00196981e+00,  4.96491849e-01, -3.59380007e-01,  4.14011553e-02, -3.13198626e-01, -2.96995908e-01,  5.94891846e-01, -4.33516592e-01,
     1.23613305e-01,  9.88423944e-01,  5.44014156e-01, -4.14300889e-01,  1.51270702e-01, -5.15383072e-02, -8.53323817e-01, -3.79216015e-01},
   {-9.20118466e-02,  6.79630637e-01,  1.42661370e-02,  1.07125151e+00, -1.95196360e-01,  7.07198143e-01,  6.88384116e-01,  1.21519184e+00,
    -7.10107327e-01, -3.08550745e-01, -4.43318546e-01,  3.32783431e-01,  7.57453740e-01,  8.49550188e-01,  2.00832888e-01, -5.01706183e-01},
   { 1.24431646e+00,  2.71933150e+00,  1.53787255e+00,  4.29266095e-01,  1.05640090e+00, -9.58783150e-01, -8.99111748e-01, -1.81875575e+00,
    -1.41779184e+00,  1.94132566e+00,  1.84840292e-01, -3.57623398e-01, -5.03833532e-01,  2.61762834e+00,  1.94203377e+00,  8.20572257e-01},
   {-4.29274112e-01,  1.49642885e+00,  3.31037968e-01,  4.00832862e-01, -6.94545638e-03,  9.55706239e-01, -8.59542072e-01, -1.58373654e+00,
    -1.95485699e+00,  1.56801498e+00,  9.15003657e-01,  1.69491518e+00, -5.81819236e-01,  7.29098201e-01,  1.87534070e+00,  2.77856854e-03},
   {-5.87236136e-05, -9.28663135e-01, -2.03842545e+00,  3.80553603e-01, -5.10141671e-01,  8.06563914e-01, -2.86439610e+00, -1.63898468e+00,
    -3.62251461e-01,  1.00076890e+00,  1.72906911e+00, -1.01015615e+00, -6.55336499e-01, -2.36975476e-01, -1.63075089e+00,  7.87749648e-01},
};

const float GRNN1_U1[8][32] = {
   {-1.01324165e+00, -1.83666438e-01, -4.61566448e-01,  6.61306977e-02, -1.15540370e-01,  1.65526897e-01, -8.00926864e-01, -1.28851026e-01,
     1.40190089e+00,  5.79985678e-01, -2.22005956e-02, -5.38649619e-01,  1.77907601e-01, -2.16669774e+00, -1.80295599e+00,  7.09731936e-01,
     1.28290176e+00, -9.70336378e-01, -3.27127606e-01, -7.74582505e-01,  3.04848742e+00,  1.27967012e+00,  8.82619023e-01, -1.52093232e+00,
    -3.24612230e-01, -8.99754226e-01,  3.47343773e-01, -4.35327828e-01,  1.51510000e-01, -1.38181329e+00, -1.44212425e+00, -2.35085517e-01},
   {-5.24785578e-01, -9.75657642e-01,  2.90603023e-02,  1.50713682e+00,  1.05516517e+00, -7.11676925e-02, -2.13409558e-01, -9.37786639e-01,
     3.81131709e-01,  1.92096746e+00, -2.05519527e-01,  6.10978067e-01, -6.13027394e-01, -1.53333998e+00, -5.50787210e-01,  5.21097183e-01,
     1.20388258e+00,  1.67043245e+00,  3.76109064e-01, -2.40365529e+00,  1.91129923e+00,  7.31996119e-01, -2.13746741e-01, -3.11830878e+00,
    -1.05485868e+00, -1.06611088e-01,  8.36064875e-01, -3.89417112e-02,  2.00135484e-01,  8.46293330e-01,  1.06796241e+00, -1.09267503e-01},
   {-3.98802578e-01, -9.27614272e-01,  1.01378524e+00, -7.99081504e-01, -3.32042924e-03,  7.81770170e-01, -4.10086960e-01,  2.12482542e-01,
     9.72548485e-01, -7.92795122e-01, -1.08619249e+00,  3.28516126e-01, -1.09537087e-01, -6.25619650e-01, -9.96973395e-01, -1.26575172e+00,
     1.49610901e+00, -4.44132924e-01,  1.36638558e+00,  1.36550203e-01,  9.14964303e-02,  1.48255199e-01, -1.77043915e+00,  2.52960420e+00,
     3.72198880e-01, -9.90201473e-01,  8.02981317e-01,  2.63297379e-01, -5.30098319e-01, -3.61288279e-01, -1.64386308e+00,  6.84758782e-01},
   {-5.27587533e-01, -3.24008912e-01, -3.68858159e-01,  4.25826669e-01, -1.61799324e+00,  3.80618334e-01,  2.17954561e-01,  2.60716421e-03,
     1.64029646e+00, -2.24472976e+00,  7.37406552e-01,  1.01024121e-01,  6.38044357e-01,  2.09620953e-01,  1.81039721e-01,  3.28173965e-01,
    -1.17441630e+00, -9.24860001e-01,  7.92309225e-01,  8.07998598e-01,  2.76109171e+00,  1.04245520e+00,  2.21775070e-01, -8.98260295e-01,
     2.04386377e+00,  9.13641632e-01, -3.41178209e-01, -2.56158501e-01, -1.82971328e-01, -1.11661327e+00,  4.50214773e-01, -7.15592325e-01},
   {-1.14364929e-01,  1.05314003e-02, -4.16911930e-01,  9.88469496e-02,  2.15452522e-01, -2.35506654e-01, -1.25379956e+00, -2.17232630e-01,
    -5.84109485e-01, -1.47166097e+00,  1.62096858e-01, -8.10365558e-01, -2.79278785e-01,  1.57691979e+00, -5.30051403e-02,  8.44115376e-01,
    -9.67475474e-01,  8.12303841e-01, -7.81137824e-01, -1.27306080e+00, -2.65087581e+00, -4.32462305e-01,  7.04908967e-01,  2.70073324e-01,
     3.12873900e-01, -4.49468017e-01, -4.57838982e-01,  2.56741136e-01,  6.81974173e-01,  1.22759962e+00,  1.12762058e+00,  1.75579011e+00},
   { 2.82424033e-01,  8.83714736e-01, -5.11964440e-01, -1.06752002e+00,  9.34016049e-01, -6.67022705e-01, -3.73367637e-01, -1.41461685e-01,
     9.21730578e-01, -1.49809837e+00,  8.58845711e-01,  4.01429743e-01, -7.08560050e-02, -8.71767342e-01, -4.43309903e-01,  8.85862350e-01,
    -1.49415624e+00, -1.28819251e+00,  1.14631927e+00,  6.27801657e-01,  3.40666223e+00,  8.21435571e-01, -1.21792269e+00,  2.80840427e-01,
    -8.85962784e-01,  1.09679949e+00, -5.77871688e-03, -3.15952092e-01,  4.63621438e-01,  1.70418561e-01,  9.83465075e-01,  1.44486308e+00},
   { 1.13165605e+00, -5.59283793e-01, -1.56179821e+00, -8.22250992e-02,  1.47727644e+00, -6.52998835e-02, -4.91240650e-01, -1.26043200e+00,
     1.22327119e-01, -1.45914912e+00,  3.48314762e-01,  8.19334090e-01,  4.56693053e-01,  3.91949594e-01, -1.31638193e+00, -1.70056760e-01,
    -1.56140220e+00,  1.11201024e+00,  1.52761424e+00,  2.87771553e-01,  5.55531919e-01,  2.36902982e-01, -1.78092480e-01, -2.55474914e-02,
    -1.81938604e-01,  5.30279040e-01, -6.83286786e-01, -3.40671577e-02, -6.24435246e-01, -2.25844949e-01, -1.93595260e-01,  4.99748439e-01},
   {-1.67011246e-01, -5.92788458e-01, -1.12752013e-01,  3.86426032e-01, -5.79899073e-01, -6.29150629e-01,  8.73395354e-02,  2.67876480e-02,
    -8.25961977e-02, -3.19994301e-01,  1.50246882e+00, -7.30119109e-01,  6.61454976e-01,  7.64915764e-01,  3.85201395e-01, -4.14835453e-01,
    -7.62227952e-01,  1.09393820e-01,  1.21528971e+00,  2.23382592e-01, -1.08678371e-01, -5.44921637e-01,  4.25901413e-01,  3.88692468e-01,
     4.09641489e-02,  4.61432427e-01, -1.36829364e+00, -1.23519468e+00, -3.73590142e-01,  1.52421546e+00,  3.38740885e-01, -4.05949682e-01},
};

const float GRNN1_U2[32][8] = {
   { 9.07552481e-01,  1.75825968e-01,  8.87061715e-01,  6.75115824e-01, -2.94081479e-01,  6.33258760e-01, -9.35505703e-02,  4.86848205e-02},
   {-7.16360748e-01,  3.04478645e-01,  1.39416397e+00,  3.10545623e-01, -7.90695667e-01,  4.60646488e-02,  2.56547600e-01,  6.94310665e-01},
   { 2.03279519e+00, -8.69742870e-01,  1.93228197e+00,  1.41436291e+00, -7.36405075e-01,  9.95498538e-01,  2.88324028e-01, -3.84684950e-01},
   { 1.67924002e-01, -3.72070909e-01, -2.55387336e-01,  5.72007358e-01, -1.96039036e-01,  4.24020678e-01,  5.33878744e-01,  1.79434359e-01},
   {-9.34239089e-01,  3.26484770e-01,  4.57318693e-01, -2.09216654e-01, -7.91335344e-01,  8.32848251e-03,  1.45826459e+00,  1.78292856e-01},
   {-4.48225796e-01, -5.95940530e-01,  1.23445714e+00, -3.65248561e-01,  1.00653522e-01, -9.30516124e-02,  8.04291129e-01, -5.39112508e-01},
   {-3.60422045e-01, -6.07746840e-01,  5.20173907e-02,  9.37340558e-01,  1.26536623e-01, -1.06201756e+00, -1.56814480e+00,  3.57148385e+00},
   { 2.18903482e-01,  6.05702341e-01,  4.42865252e-01, -7.69158065e-01, -2.51205295e-01, -4.46829736e-01,  1.18786387e-01, -6.89944923e-01},
   {-1.54211259e+00, -3.35325718e-01,  2.79460406e+00,  3.07851481e+00,  2.71754891e-01, -1.17077982e+00, -2.23488593e+00, -9.93798554e-01},
   {-6.87620640e-01,  6.01272807e-02, -6.01416469e-01, -5.07458031e-01, -2.91765362e-01,  4.78311956e-01, -3.48815560e-01, -2.12261453e-01},
   {-7.09160686e-01, -1.62148917e+00,  4.25562300e-02, -4.84843493e-01, -2.54487544e-01,  5.48061967e-01, -1.75715059e-01, -4.04285163e-01},
   { 4.53788400e-01,  5.47231376e-01, -7.83877254e-01, -1.45531380e+00,  6.24865472e-01,  4.97991502e-01,  4.73094761e-01, -3.59340101e-01},
   { 1.11636221e+00, -1.62115979e+00,  9.10427272e-02, -7.94163167e-01,  6.11717582e-01, -1.70128858e+00, -1.01068163e+00, -2.39055946e-01},
   {-3.14121604e-01, -9.27745044e-01,  6.67161345e-01, -1.13658309e+00,  4.39912617e-01,  5.97238064e-01,  2.67864764e-01,  9.57389414e-01},
   {-5.14824808e-01,  2.33282328e-01, -1.90341783e+00, -3.08007985e-01,  2.81464887e+00, -4.92268592e-01, -1.51766276e+00,  1.62734091e+00},
   {-4.84787107e-01, -8.08812380e-01,  2.79133767e-01,  1.09706962e+00, -7.07464099e-01, -1.68869269e+00, -2.02725744e+00,  2.57951260e+00},
   { 1.03210914e+00, -1.63552356e+00,  1.34389997e+00,  3.16669524e-01,  1.10978091e+00, -2.62642217e+00, -1.01516736e+00,  4.66321498e-01},
   {-1.72011125e+00,  2.16430664e+00,  1.23684025e+00,  2.83450794e+00,  1.19767070e+00, -5.22092760e-01, -1.22539890e+00,  1.42188394e+00},
   {-2.01423001e+00, -1.12397110e+00,  1.34478319e+00, -1.93276107e-01,  1.13203204e+00,  1.09572518e+00,  1.86988544e+00,  7.66349077e-01},
   { 2.60749072e-01, -1.45853639e+00,  3.52641535e+00,  1.18246162e+00, -3.48595738e-01, -1.11231804e+00, -1.24223948e+00,  5.74731687e-03},
   {-5.17732538e-02, -7.76420891e-01, -1.34966075e-01,  8.59457478e-02, -2.02159226e-01,  6.77571535e-01, -5.19744456e-01, -4.41278934e-01},
   {-2.46717072e+00,  2.02927971e+00,  2.47370332e-01,  2.06183481e+00,  6.26464427e-01, -7.16500700e-01,  4.88436580e-01,  8.48021567e-01},
   {-5.03125489e-01, -1.10484409e+00, -6.73429489e-01, -1.30532086e+00, -3.33634019e-01,  3.59124988e-01,  1.87715963e-01, -2.22651005e-01},
   {-2.05462384e+00, -1.77640283e+00, -7.95803130e-01,  5.09055018e-01,  3.00927949e+00, -1.85185587e+00,  4.13955688e-01, -5.93347609e-01},
   {-3.17620724e-01,  4.38309491e-01,  9.68974158e-02,  1.41030777e+00,  4.66682255e-01, -1.06963122e+00, -8.20571005e-01,  2.09842920e+00},
   {-6.21467471e-01,  1.81829083e+00, -6.12469733e-01,  1.60692465e+00,  2.04890132e+00, -2.48222295e-02, -1.14864564e+00,  2.51117754e+00},
   { 3.94487828e-01,  7.20217377e-02,  1.52702296e+00,  3.15189101e-02, -2.08630061e+00, -1.39857304e+00, -5.76356411e-01,  9.49095488e-01},
   {-3.82879287e-01,  1.03405941e+00,  1.45499378e-01,  1.15501130e+00, -5.96912503e-01,  7.81852543e-01,  1.17504025e+00,  4.60816234e-01},
   { 4.74841088e-01, -4.91103828e-01,  1.82344824e-01, -9.82399564e-03, -1.10781126e-01,  3.13602686e-01,  7.28638321e-02,  1.00032613e-01},
   { 6.42354071e-01, -3.09788078e-01,  1.16575694e+00,  3.44104230e-01,  1.15355825e+00,  1.91826582e+00, -1.33364761e+00, -3.01620811e-01},
   { 1.91467297e+00,  1.62566197e+00,  9.50617194e-01, -1.14855564e+00,  1.17193449e+00,  6.45478904e-01, -5.08744836e-01,  2.01746893e+00},
   {-1.86488855e+00,  8.26546550e-01,  2.36057901e+00,  1.95642161e+00,  9.14012790e-01,  4.01838094e-01, -4.22858536e-01,  2.03237033e+00},
};

// clang-format on


#endif // __FASTGRNN_GRNN1_PARAMS__

