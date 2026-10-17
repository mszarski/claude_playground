"""LM7805 12 V -> 5 V with the datasheet 330n / 100n capacitors and a 1k-limited green power LED."""
vin, v5, gnd, la = Net("VIN"), Net("+5V"), Net("GND"), Net("LED_A")
vin.drive = gnd.drive = POWER
header(vin, gnd); header(v5, gnd)
u = Part("Regulator_Linear", "LM7805_TO220", footprint="Package_TO_SOT_THT:TO-220-3_Vertical")
u["VI"] += vin; u["GND"] += gnd; u["VO"] += v5
vin & C("330n") & gnd
v5 & C("100n") & gnd
led = Part("Device", "LED", value="green", footprint="LED_SMD:LED_0805_2012Metric")
v5 & R("1k") & la
la += led["A"]; led["K"] += gnd
