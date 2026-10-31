"""Window comparator on 5 V: a 20k/10k/20k ladder sets 3 V and 2 V. U1A is high while
IN < 3 V, U1B while IN > 2 V; diodes to each output pull OUT low unless both are high."""
vcc, gnd, inp, out = Net("VCC"), Net("GND"), Net("IN"), Net("OUT")
vh, vl, oa, ob = Net("VH"), Net("VL"), Net("OA"), Net("OB")
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(inp, gnd); header(out, gnd)
u = Part("Amplifier_Operational", "LM358", footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
u["V+"] += vcc; u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("20k", tolerance=0.01) & vh & R("10k", tolerance=0.01) & vl & R("20k", tolerance=0.01) & gnd
u[3] += vh; u[2] += inp; u[1] += oa
u[5] += inp; u[6] += vl; u[7] += ob
vcc & R("10k") & out
for o in (oa, ob):
    d = Part("Diode", "1N4148W", footprint="Diode_SMD:D_SOD-123")
    d["A"] += out; d["K"] += o
