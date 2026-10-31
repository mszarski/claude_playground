"""Balanced line receiver on +/-12 V: unity-gain difference amplifier (10k, 0.1%) then a
unity-gain Sallen-Key Butterworth low-pass at ~20 kHz (R = 5.36k, C1 = 2.2n feedback,
C2 = 1n to GND) as the output stage."""
vpos, vneg, gnd = Net("VPOS"), Net("VNEG"), Net("GND")
inp, inn, out = Net("IN_P"), Net("IN_N"), Net("OUT")
ni, inv, d, m, x = Net("NI"), Net("INV"), Net("D"), Net("M"), Net("X")
vpos.drive = vneg.drive = gnd.drive = POWER
header(vpos, gnd, vneg); header(inp, inn, gnd); header(out, gnd)
u = tl072()
u["V+"] += vpos; u["V-"] += vneg
vpos & C("100n") & gnd; vneg & C("100n") & gnd
inn & R("10k", tolerance=0.001) & inv & R("10k", tolerance=0.001) & d
inp & R("10k", tolerance=0.001) & ni & R("10k", tolerance=0.001) & gnd
u[2] += inv; u[3] += ni; u[1] += d
d & R("5.36k", tolerance=0.01) & m & R("5.36k", tolerance=0.01) & x
m & C("2.2n") & out
x & C("1n") & gnd
u[5] += x; u[6] += out; u[7] += out
