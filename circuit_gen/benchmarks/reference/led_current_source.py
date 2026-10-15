"""Two-NPN constant-current sink: I_LED ~ Vbe / 33 R ~ 20 mA, independent of VIN (9-15 V)."""
vin, gnd, k, b1, e1 = Net("VIN"), Net("GND"), Net("LED_K"), Net("B1"), Net("E1")
vin.drive = gnd.drive = POWER
header(vin, gnd)
led = Part("Device", "LED", value="white", footprint="LED_SMD:LED_0805_2012Metric")
q1 = Part("Transistor_BJT", "2N3904", footprint="Package_TO_SOT_THT:TO-92_Inline")
q2 = Part("Transistor_BJT", "2N3904", footprint="Package_TO_SOT_THT:TO-92_Inline")
vin += led["A"]; led["K"] += k
q1["C"] += k; q1["B"] += b1; q1["E"] += e1
vin & R("4.7k") & b1
q2["C"] += b1; q2["B"] += e1; q2["E"] += gnd
e1 & R("33") & gnd
vin & C("100n") & gnd
