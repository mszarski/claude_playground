"""Virtual ground from a 9 V battery: 10k/10k divider with 1u filter, TL072 follower drives
VMID (feedback taken at VMID), 10u on VMID; the unused section is a follower of the divider."""
vbat, gnd, vmid, div = Net("VBAT"), Net("GND"), Net("VMID"), Net("DIV")
vbat.drive = gnd.drive = POWER
header(vbat, vmid, gnd)
u = tl072()
u["V+"] += vbat; u["V-"] += gnd
vbat & C("100n") & gnd
vbat & R("10k", tolerance=0.01) & div & R("10k", tolerance=0.01) & gnd
div & C("1u") & gnd
u[3] += div; u[2] += vmid; u[1] += vmid
vmid & C("10u") & gnd
u[5] += div; u[6] += u[7]
