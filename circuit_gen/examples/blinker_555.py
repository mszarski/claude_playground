"""NE555 astable LED blinker, about 1 Hz, powered from 5-12 V."""

vcc, gnd = Net("VCC"), Net("GND")
vcc.drive = gnd.drive = POWER  # supplied through the power connector
dis, thr, out, cv, led_a = Net("DIS"), Net("THR"), Net("OUT"), Net("CV"), Net("LED_A")

R = lambda value: Part("Device", "R", value=value, footprint="Resistor_SMD:R_0805_2012Metric")
C = lambda value: Part("Device", "C", value=value, footprint="Capacitor_SMD:C_0805_2012Metric")

j_pwr = Part("Connector", "Conn_01x02_Pin",
             footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical")
timer = Part("Timer", "NE555D", footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
led = Part("Device", "LED", value="red", footprint="LED_SMD:LED_0805_2012Metric")

j_pwr[1, 2] += vcc, gnd
timer["VCC"] += vcc
timer["R"] += vcc
timer["GND"] += gnd
timer["DIS"] += dis
timer["THR"] += thr
timer["TR"] += thr
timer["CV"] += cv
timer["Q"] += out

# f = 1.44 / ((R1 + 2*R2) * C) = 1.44 / ((10k + 2*68k) * 10u) ~= 1 Hz
vcc & R("10k") & dis
dis & R("68k") & thr
thr & C("10u") & gnd
cv & C("10n") & gnd
vcc & C("100n") & gnd
out & R("470") & led_a
led_a += led["A"]
led["K"] += gnd
