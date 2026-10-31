"""Two non-inverting stages of gain 1 + 30k/1k = 31 each (961 total) on 9 V, mid-supply
reference, AC coupling at the input, between stages and at the output."""
vcc, gnd, inp, out = Net("VCC"), Net("GND"), Net("IN"), Net("OUT")
ref, n1, i1, f1, o1, n2, i2, f2, o2 = (Net(n) for n in ("REF", "N1", "I1", "F1", "O1", "N2", "I2", "F2", "O2"))
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(inp, gnd); header(out, gnd)
u = tl072()
u["V+"] += vcc; u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("10k") & ref & R("10k") & gnd
ref & C("10u") & gnd
inp & C("1u") & n1 & R("100k") & ref
u[3] += n1; u[2] += i1; u[1] += o1
i1 & R("30k") & o1
i1 & R("1k") & f1 & C("10u") & gnd
o1 & C("1u") & n2 & R("100k") & ref
u[5] += n2; u[6] += i2; u[7] += o2
i2 & R("30k") & o2
i2 & R("1k") & f2 & C("10u") & gnd
o2 & C("10u") & out
