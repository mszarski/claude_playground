"""Difference amplifier, gain 100k/10k = 10, on +/-12 V."""
vpos, vneg, gnd = Net("VPOS"), Net("VNEG"), Net("GND")
inp, inn, out, ni, inv = Net("IN_P"), Net("IN_N"), Net("OUT"), Net("NI"), Net("INV")
vpos.drive = vneg.drive = gnd.drive = POWER
header(vpos, gnd, vneg); header(inp, inn, gnd); header(out, gnd)
u = tl072()
u["V+"] += vpos; u["V-"] += vneg
vpos & C("100n") & gnd; vneg & C("100n") & gnd
inn & R("10k", tolerance=0.001) & inv & R("100k", tolerance=0.001) & out
inp & R("10k", tolerance=0.001) & ni & R("100k", tolerance=0.001) & gnd
u[2] += inv; u[3] += ni; u[1] += out
u[5] += gnd; u[6] += u[7]
