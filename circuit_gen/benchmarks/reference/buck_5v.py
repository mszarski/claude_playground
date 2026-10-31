"""LM2596-5.0 typical application, 12 V in, 5 V / 1 A out: 680u input, 33uH inductor,
SS34 Schottky catch diode, 220u output, FB to the output, ON/OFF grounded (on)."""
vin, v5, gnd, sw = Net("VIN"), Net("+5V"), Net("GND"), Net("SW")
vin.drive = gnd.drive = POWER
header(vin, gnd); header(v5, gnd)
u = Part("Regulator_Switching", "LM2596S-5", footprint="Package_TO_SOT_SMD:TO-263-5_TabPin3")
u["VIN"] += vin; u["GND"] += gnd; u["OUT"] += sw; u["FB"] += v5; u[5] += gnd
cin = Part("Device", "C_Polarized", value="680u", footprint="Capacitor_SMD:CP_Elec_10x10.5")
cin[1] += vin; cin[2] += gnd
vin & C("100n") & gnd
d = Part("Diode", "SS34", footprint="Diode_SMD:D_SMA")
d["K"] += sw; d["A"] += gnd
l = Part("Device", "L", value="33u", footprint="Inductor_SMD:L_12x12mm_H8mm")
l[1] += sw; l[2] += v5
cout = Part("Device", "C_Polarized", value="220u", footprint="Capacitor_SMD:CP_Elec_10x10.5")
cout[1] += v5; cout[2] += gnd
