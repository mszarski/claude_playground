"""XOR from four NANDs: N1 = A nand B; OUT = (A nand N1) nand (B nand N1). 10k pull-downs."""
vcc, gnd, a, b, out = Net("VCC"), Net("GND"), Net("A"), Net("B"), Net("OUT")
n1, n2, n3 = Net("N1"), Net("N2"), Net("N3")
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(a, b, gnd); header(out, gnd)
u = Part("74xx", "74HC00", footprint="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm")
u["VCC"] += vcc; u["GND"] += gnd
vcc & C("100n") & gnd
a & R("10k") & gnd
b & R("10k") & gnd
u[1] += a; u[2] += b; u[3] += n1
u[4] += a; u[5] += n1; u[6] += n2
u[9] += b; u[10] += n1; u[8] += n3
u[12] += n2; u[13] += n3; u[11] += out
