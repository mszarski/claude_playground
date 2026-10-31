"""555 PWM at ~1 kHz, ~25% duty: a diode across RB separates charge (through RA) and
discharge (through RB): t_high ~ 0.693 RA C, t_low ~ 0.693 RB C, C = 100n, RA = 3k, RB = 11k."""
vcc, gnd, out, dis, thr, cv = (Net(n) for n in ("VCC", "GND", "OUT", "DIS", "THR", "CV"))
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(out, gnd)
t = Part("Timer", "NE555D", footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
t["VCC"] += vcc; t["R"] += vcc; t["GND"] += gnd
t["DIS"] += dis; t["THR"] += thr; t["TR"] += thr; t["CV"] += cv; t["Q"] += out
vcc & R("3k", tolerance=0.01) & dis & R("11k", tolerance=0.01) & thr
d = Part("Diode", "1N4148W", footprint="Diode_SMD:D_SOD-123")
d["A"] += dis; d["K"] += thr
thr & C("100n") & gnd
cv & C("10n") & gnd
vcc & C("100n") & gnd
