"""IRLZ44N logic-level low-side switch: 100 R gate resistor, 10k gate pull-down, flyback diode
across the load (cathode to V12)."""
v12, gnd, load_neg, ctrl, gate = Net("V12"), Net("GND"), Net("LOAD_NEG"), Net("CTRL"), Net("GATE")
v12.drive = gnd.drive = POWER
header(v12, gnd); header(v12, load_neg); header(ctrl, gnd)
q = Part("Transistor_FET", "IRLZ44N", footprint="Package_TO_SOT_THT:TO-220-3_Vertical")
d = Part("Diode", "1N4148W", footprint="Diode_SMD:D_SOD-123")
q["D"] += load_neg; q["S"] += gnd; q["G"] += gate
ctrl & R("100") & gate
gate & R("10k") & gnd
d["K"] += v12; d["A"] += load_neg
v12 & C("10u") & gnd
