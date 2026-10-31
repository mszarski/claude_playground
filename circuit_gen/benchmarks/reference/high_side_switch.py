"""High-side switch: AO3401A P-MOSFET, gate held at V12 by 10k; a BC847 pulls the gate down
through 10k (Vgs ~ -6 V, inside the +/-12 V rating). 10k base resistor, 100k base pull-down."""
v12, gnd, ctrl, load = Net("V12"), Net("GND"), Net("CTRL"), Net("LOAD_POS")
gate, col, base = Net("GATE"), Net("COL"), Net("BASE")
v12.drive = gnd.drive = POWER
header(v12, gnd); header(load, gnd); header(ctrl, gnd)
q = Part("Transistor_FET", "AO3401A", footprint="Package_TO_SOT_SMD:SOT-23")
q["S"] += v12; q["D"] += load; q["G"] += gate
v12 & R("10k") & gate & R("10k") & col
n = Part("Transistor_BJT", "BC847", footprint="Package_TO_SOT_SMD:SOT-23")
n["C"] += col; n["E"] += gnd; n["B"] += base
ctrl & R("10k") & base & R("100k") & gnd
v12 & C("10u") & gnd
