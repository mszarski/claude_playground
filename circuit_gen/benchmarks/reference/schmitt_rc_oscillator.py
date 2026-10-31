"""74HC14 RC oscillator: one Schmitt inverter with 11k feedback and 100n to GND,
f ~ 1 / (0.91 R C) ~ 1 kHz with thresholds near 0.55 and 0.33 Vcc; a second inverter
buffers the output. Unused inputs to GND."""
vcc, gnd, out, a, y = Net("VCC"), Net("GND"), Net("OUT"), Net("A"), Net("Y")
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(out, gnd)
u = Part("74xx", "74HC14", footprint="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm")
u["VCC"] += vcc; u["GND"] += gnd
vcc & C("100n") & gnd
u[1] += a; u[2] += y
y & R("11k", tolerance=0.01) & a
a & C("100n") & gnd
u[3] += y; u[4] += out
for i in (5, 9, 11, 13):
    u[i] += gnd
