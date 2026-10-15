# Shared helpers for reference designs (prepended to each design by the test).
R = lambda v, **kw: Part("Device", "R", value=v, footprint="Resistor_SMD:R_0805_2012Metric", **kw)
C = lambda v: Part("Device", "C", value=v, footprint="Capacitor_SMD:C_0805_2012Metric")
def header(*nets):
    j = Part("Connector", f"Conn_01x{len(nets):02d}_Pin",
             footprint=f"Connector_PinHeader_2.54mm:PinHeader_1x{len(nets):02d}_P2.54mm_Vertical")
    for i, n in enumerate(nets, 1):
        j[i] += n
    return j
def tl072():
    return Part("Amplifier_Operational", "TL072", footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
