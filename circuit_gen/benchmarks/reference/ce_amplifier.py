"""NPN common-emitter amplifier on 9 V: Vb ~ 1.6 V, Ic ~ 1 mA, gain ~ (4.7k || 10k) / (re + 120) ~ 20."""
vcc, gnd, inp, out = Net("VCC"), Net("GND"), Net("IN"), Net("OUT")
base, col, em, em2 = Net("BASE"), Net("COL"), Net("EM"), Net("EM2")
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(inp, gnd); header(out, gnd)
q = Part("Transistor_BJT", "2N3904", footprint="Package_TO_SOT_THT:TO-92_Inline")
vcc & R("47k") & base & R("10k") & gnd
vcc & R("4.7k") & col
q["B"] += base; q["C"] += col; q["E"] += em
em & R("120") & em2 & R("820") & gnd
em2 & C("100u") & gnd
inp & C("10u") & base
col & C("10u") & out
vcc & C("100n") & gnd
