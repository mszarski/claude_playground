"""P-MOSFET reverse-polarity protection: drain at the input, source at the output, gate pulled to GND.
A 10 V zener from source to gate keeps Vgs inside the AO3401A's +/-12 V rating."""
vin, vout, gnd, g = Net("VIN"), Net("VOUT"), Net("GND"), Net("GATE")
vin.drive = gnd.drive = POWER
header(vin, gnd); header(vout, gnd)
q = Part("Transistor_FET", "AO3401A", footprint="Package_TO_SOT_SMD:SOT-23")
q["D"] += vin; q["S"] += vout; q["G"] += g
g & R("10k") & gnd
z = Part("Device", "D_Zener", value="10V", footprint="Diode_SMD:D_SOD-123")
z["K"] += vout; z["A"] += g
vout & C("10u") & gnd
