"""555 astable, f = 1.44 / ((Ra + 2 Rb) C) = 1.44 / ((10k + 2 * 68k) * 10u) ~ 0.99 Hz.
LED1 lights when OUT is high, LED2 (from VCC) when OUT is low."""
vcc, gnd, out, dis, thr, cv = (Net(n) for n in ("VCC", "GND", "OUT", "DIS", "THR", "CV"))
vcc.drive = gnd.drive = POWER
header(vcc, gnd)
t = Part("Timer", "NE555D", footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
t["VCC"] += vcc; t["R"] += vcc; t["GND"] += gnd
t["DIS"] += dis; t["THR"] += thr; t["TR"] += thr; t["CV"] += cv; t["Q"] += out
vcc & R("10k") & dis & R("68k") & thr
thr & C("10u") & gnd
cv & C("10n") & gnd
vcc & C("100n") & gnd
l1 = Part("Device", "LED", value="red", footprint="LED_SMD:LED_0805_2012Metric")
l2 = Part("Device", "LED", value="green", footprint="LED_SMD:LED_0805_2012Metric")
a1, a2 = Net("A1"), Net("K2")
out & R("220") & a1; a1 += l1["A"]; l1["K"] += gnd
vcc += l2["A"]; l2["K"] += a2; a2 & R("220") & out
