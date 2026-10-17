"""2.5 V reference: 10k/10k divider (1%) on 5 V, 10u filter, LM358 voltage follower."""
vcc, gnd, out, div = Net("VCC"), Net("GND"), Net("VREF_OUT"), Net("DIV")
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(out, gnd)
u = Part("Amplifier_Operational", "LM358", footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
u["V+"] += vcc; u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("10k", tolerance=0.01) & div & R("10k", tolerance=0.01) & gnd
div & C("10u") & gnd
u[3] += div; u[2] += out; u[1] += out
u[5] += div; u[6] += u[7]          # unused section as a follower
