"""Three-input inverting summer on 9 V: each input through 10u and 10k into the
inverting node, 10k feedback (gain -1 per input), non-inverting input at a 10k/10k
mid-supply reference with 10u, output coupled through 10u."""
vcc, gnd, out, ref, inv, oa = Net("VCC"), Net("GND"), Net("OUT"), Net("REF"), Net("INV"), Net("OA")
vcc.drive = gnd.drive = POWER
ins = [Net(f"IN{k}") for k in (1, 2, 3)]
header(vcc, gnd); header(*ins, gnd); header(out, gnd)
u = tl072()
u["V+"] += vcc; u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("10k") & ref & R("10k") & gnd
ref & C("10u") & gnd
for k, n in enumerate(ins, 1):
    c = Net(f"C{k}")
    n & C("10u") & c & R("10k", tolerance=0.01) & inv
inv & R("10k", tolerance=0.01) & oa
u[2] += inv; u[3] += ref; u[1] += oa
oa & C("10u") & out
u[5] += ref; u[6] += u[7]
