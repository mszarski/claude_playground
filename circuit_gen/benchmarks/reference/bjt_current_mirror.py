"""NPN mirror sinking ~1 mA: Iref = (12 - 0.65) / (11k + 470) ~ 0.99 mA, with 470 R emitter
degeneration in both legs to raise the output resistance (Early effect)."""
vcc, gnd, out, ref, e1, e2 = Net("VCC"), Net("GND"), Net("OUT"), Net("REF"), Net("E1"), Net("E2")
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(out, gnd)
q1 = Part("Transistor_BJT", "BC847", footprint="Package_TO_SOT_SMD:SOT-23")
q2 = Part("Transistor_BJT", "BC847", footprint="Package_TO_SOT_SMD:SOT-23")
vcc & R("11k", tolerance=0.01) & ref
q1["C"] += ref; q1["B"] += ref; q1["E"] += e1
q2["C"] += out; q2["B"] += ref; q2["E"] += e2
e1 & R("470", tolerance=0.01) & gnd
e2 & R("470", tolerance=0.01) & gnd
vcc & C("100n") & gnd
