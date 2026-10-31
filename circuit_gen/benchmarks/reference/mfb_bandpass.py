"""Multiple-feedback band-pass, f0 = 1 kHz, Q = 2, centre gain 1, C = 10n:
R1 = Q/(H0 w C) = 31.6k, R2 = Q/((2Q^2 - H0) w C) = 4.53k, R3 = 2Q/(w C) = 63.4k (1%)."""
vpos, vneg, gnd, inp, out = Net("VPOS"), Net("VNEG"), Net("GND"), Net("IN"), Net("OUT")
x, inv = Net("X"), Net("INV")
vpos.drive = vneg.drive = gnd.drive = POWER
header(vpos, gnd, vneg); header(inp, gnd); header(out, gnd)
u = tl072()
u["V+"] += vpos; u["V-"] += vneg
vpos & C("100n") & gnd; vneg & C("100n") & gnd
inp & R("31.6k", tolerance=0.01) & x
x & R("4.53k", tolerance=0.01) & gnd
x & C("10n") & out
x & C("10n") & inv
inv & R("63.4k", tolerance=0.01) & out
u[2] += inv; u[3] += gnd; u[1] += out
u[5] += gnd; u[6] += u[7]
