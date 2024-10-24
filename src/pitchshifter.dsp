import("stdfaust.lib");

fdelay(n,d,x) = de.delay(n+1,int(d),x)*(1 - ma.frac(d)) + de.delay(n+1,int(d)+1,x)*ma.frac(d);

transpose(w, x, s, sig) = fdelay(maxDelay,d,sig)*ma.fmin(d/x,1) +
    fdelay(maxDelay,d+w,sig)*(1-ma.fmin(d/x,1))
with {
    maxDelay = 32768 / 2;
    i = 1 - pow(2, s/12);
    d = i : (+ : +(w) : fmod(_,w)) ~ _;
};

looper = (+) ~ (*(0.3) : de.delay(4800, 4800));

process =  transpose(1000, 500, hslider("semi", 0, -12, 12, 1)) <: (_, looper) : select2(button("echo"));
