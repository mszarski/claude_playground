"""MAX485 at 5 V with 120 R termination across A-B, 100n decoupling, UART and bus headers."""
vcc, gnd, a, b = Net("VCC"), Net("GND"), Net("A"), Net("B")
ro, di, dere = Net("RO"), Net("DI"), Net("DE_RE")
vcc.drive = gnd.drive = POWER
u = Part("Interface_UART", "MAX485E", footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
u["VCC"] += vcc; u["GND"] += gnd
u["RO"] += ro; u["DI"] += di; u[2] += dere; u["DE"] += dere
u["A"] += a; u["B"] += b
a & R("120") & b
vcc & C("100n") & gnd
header(ro, di, dere, vcc, gnd)
header(a, b, gnd)
