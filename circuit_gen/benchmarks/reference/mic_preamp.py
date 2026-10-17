"""Electret preamp: 2.2k mic bias, 1u input coupling into a mid-supply-biased non-inverting
stage, gain 1 + 100k/1k = 101 (1k returned to GND through 10u so the DC gain is 1), 10u output coupling."""
vcc, gnd, mic, out = Net("VCC"), Net("GND"), Net("MIC"), Net("AUDIO_OUT")
vref, ninp, inv, fb, oa = Net("VREF"), Net("NINP"), Net("INV"), Net("FB"), Net("OA")
vcc.drive = gnd.drive = POWER
header(vcc, gnd); header(mic, gnd); header(out, gnd)
u = Part("Amplifier_Operational", "LM358", footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
u["V+"] += vcc; u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("2.2k") & mic
vcc & R("100k") & vref & R("100k") & gnd
vref & C("10u") & gnd
mic & C("1u") & ninp & R("100k") & vref
u[3] += ninp; u[2] += inv; u[1] += oa
inv & R("100k") & oa
inv & R("1k") & fb & C("10u") & gnd
oa & C("10u") & out
u[5] += vref; u[6] += u[7]
