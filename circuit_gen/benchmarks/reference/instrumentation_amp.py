"""Three-op-amp instrumentation amplifier, gain (1 + 2*10k/2.2k) * (100k/10k) = 101, +/-12 V.
Input buffers give high input impedance; 0.1% resistors in the difference stage for CMRR."""
vpos, vneg, gnd = Net("VPOS"), Net("VNEG"), Net("GND")
inp, inn, out = Net("IN_P"), Net("IN_N"), Net("OUT")
a1, a2, g1, g2, ni, inv = Net("A1"), Net("A2"), Net("G1"), Net("G2"), Net("NI"), Net("INV")
vpos.drive = vneg.drive = gnd.drive = POWER
header(vpos, gnd, vneg); header(inp, inn, gnd); header(out, gnd)
u1, u2 = tl072(), tl072()
for u in (u1, u2):
    u["V+"] += vpos; u["V-"] += vneg
vpos & C("100n") & gnd; vneg & C("100n") & gnd
u1[3] += inp; u1[2] += g1; u1[1] += a1           # input buffer, + side
u1[5] += inn; u1[6] += g2; u1[7] += a2           # input buffer, - side
a1 & R("10k", tolerance=0.01) & g1 & R("2.2k", tolerance=0.01) & g2 & R("10k", tolerance=0.01) & a2
a2 & R("10k", tolerance=0.001) & inv & R("100k", tolerance=0.001) & out
a1 & R("10k", tolerance=0.001) & ni & R("100k", tolerance=0.001) & gnd
u2[2] += inv; u2[3] += ni; u2[1] += out
u2[5] += gnd; u2[6] += u2[7]                     # unused section as a follower
