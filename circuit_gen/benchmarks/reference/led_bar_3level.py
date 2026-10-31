"""Three-LED bar graph on 5 V: a 20k/10k/10k/10k ladder sets 3, 2 and 1 V; LM324 sections
compare IN with each and drive a red LED through 330 R (~9 mA) when IN is above it.
LED1..LED3 are the anodes of the 1 V, 2 V and 3 V LEDs."""
vcc, gnd, inp = Net("VCC"), Net("GND"), Net("IN")
t3, t2, t1 = Net("T3"), Net("T2"), Net("T1")
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(inp, gnd)
u = Part("Amplifier_Operational", "LM324", footprint="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm")
u["V+"] += vcc; u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("20k", tolerance=0.01) & t3 & R("10k", tolerance=0.01) & t2 & R("10k", tolerance=0.01) & t1 & R("10k", tolerance=0.01) & gnd
for (plus, minus, outp), thr, name in zip(((3, 2, 1), (5, 6, 7), (10, 9, 8)), (t1, t2, t3),
                                        ("LED1", "LED2", "LED3")):
    o, la = Net(), Net(name)
    u[plus] += inp; u[minus] += thr; u[outp] += o
    led = Part("Device", "LED", value="red", footprint="LED_SMD:LED_0805_2012Metric")
    o & R("330") & la
    la += led["A"]; led["K"] += gnd
u[12] += gnd; u[13] += u[14]                      # unused section as a follower of GND
