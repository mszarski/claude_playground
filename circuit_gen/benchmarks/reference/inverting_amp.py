"""Single-supply inverting amplifier, gain -100k/10k = -10, AC-coupled, mid-supply reference."""
vcc, gnd, inp, out = Net("VCC"), Net("GND"), Net("IN"), Net("OUT")
vref, cin, inv, oa = Net("VREF"), Net("CIN"), Net("INV"), Net("OA")
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(inp, gnd); header(out, gnd)
u = Part("Amplifier_Operational", "LM358", footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
u["V+"] += vcc; u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("10k") & vref & R("10k") & gnd
vref & C("10u") & gnd
inp & C("10u") & cin & R("10k") & inv
inv & R("100k") & oa
u[2] += inv; u[3] += vref; u[1] += oa
oa & C("10u") & out
u[5] += vref; u[6] += u[7]
