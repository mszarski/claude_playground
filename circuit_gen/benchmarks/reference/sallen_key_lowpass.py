"""Unity-gain Sallen-Key Butterworth low-pass, fc = 1/(2*pi*R*sqrt(C1*C2)) ~ 1.02 kHz, Q = 0.707."""
vpos, vneg, gnd, inp, out, a, b = (Net(n) for n in ("VPOS", "VNEG", "GND", "IN", "OUT", "A", "B"))
vpos.drive = vneg.drive = gnd.drive = POWER
header(vpos, gnd, vneg); header(inp, gnd); header(out, gnd)
u = tl072()
u["V+"] += vpos; u["V-"] += vneg
vpos & C("100n") & gnd; vneg & C("100n") & gnd
inp & R("10k") & a & R("10k") & b
a & C("22n") & out          # C1, feedback
b & C("11n") & gnd          # C2, to ground
u[3] += b; u[2] += out; u[1] += out
u[5] += gnd; u[6] += u[7]   # unused section as a follower at ground
