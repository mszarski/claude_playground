"""Button to GND with a 10k pull-up, 10k/1u RC (~10 ms), then a Schmitt inverter:
released -> input high -> OUT low; pressed -> input low -> OUT high."""
vcc, gnd, out, raw, din = Net("VCC"), Net("GND"), Net("OUT"), Net("RAW"), Net("DIN")
vcc.drive = gnd.drive = POWER
header(vcc, gnd, out)
sw = Part("Switch", "SW_Push", footprint="Button_Switch_SMD:SW_SPST_PTS645")
u = Part("74xx", "74HC14", footprint="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm")
vcc & R("10k") & raw
sw[1] += raw; sw[2] += gnd
raw & R("10k") & din
din & C("1u") & gnd
u[1] += din; u[2] += out
for i in (3, 5, 9, 11, 13):          # unused inputs to GND
    u[i] += gnd
u["VCC"] += vcc; u["GND"] += gnd
vcc & C("100n") & gnd
