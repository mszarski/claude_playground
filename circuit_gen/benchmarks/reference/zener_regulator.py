"""Zener shunt regulator: Rs = (12 - 5.1) / (10 mA load + 5 mA zener) ~ 470 R."""
vin, out, gnd = Net("VIN"), Net("OUT"), Net("GND")
vin.drive = gnd.drive = POWER
header(vin, gnd); header(out, gnd)
z = Part("Device", "D_Zener", value="5.1V", footprint="Diode_SMD:D_SOD-123")
vin & R("470") & out
z["K"] += out; z["A"] += gnd
out & C("10u") & gnd
