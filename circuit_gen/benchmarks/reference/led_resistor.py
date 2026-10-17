"""Red LED (Vf ~1.9 V) at ~10 mA from 5 V: R = (5 - 1.9) / 0.01 = 310 R -> 330 R."""
vcc, gnd, a = Net("VCC"), Net("GND"), Net("LED_A")
vcc.drive = gnd.drive = POWER
header(vcc, gnd)
led = Part("Device", "LED", value="red", footprint="LED_SMD:LED_0805_2012Metric")
vcc & R("330") & a
a += led["A"]; led["K"] += gnd
