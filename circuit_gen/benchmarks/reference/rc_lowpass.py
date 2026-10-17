"""First-order RC low-pass: fc = 1 / (2 pi 1.6k 100n) = 995 Hz."""
inp, out, gnd = Net("IN"), Net("OUT"), Net("GND")
header(inp, gnd); header(out, gnd)
inp & R("1.6k", tolerance=0.01) & out
out & C("100n") & gnd
