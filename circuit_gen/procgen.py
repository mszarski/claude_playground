"""
Programmatic training data: parametric circuit families, no LLM needed.

Each family samples parameters, computes component values from the design
equations, rounds them to standard (E-series) values, and writes three
things: a board description in the benchmark's style, the SKiDL circuit with
comments showing the calculation, and require(...) lines for the behaviour.
Every generated board is then verified with the same checks used on model
output (structure, ERC, and the requirements simulated across component
tolerances); only boards that pass are kept.

This is cheap and exact but template-shaped; LLM-designed sessions add the
variety on top.
"""

import math
import random
import re

E6 = [1.0, 1.5, 2.2, 3.3, 4.7, 6.8]
E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3,
       4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]

R_FP = "Resistor_SMD:R_0805_2012Metric"
C_FP = "Capacitor_SMD:C_0805_2012Metric"
LED_FP = "LED_SMD:LED_0805_2012Metric"
SOIC8 = "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"

# Same forward voltages the simulator uses for LED colours (spice._LED_VF).
LED_VF = {"red": 1.9, "yellow": 2.0, "green": 2.2, "blue": 3.0, "white": 3.0}

OPAMPS = [("LM358", SOIC8), ("TL072", SOIC8), ("MCP6002-xSN", SOIC8)]

REGULATORS = [  # symbol, footprint, output volts, dropout volts, pin names (in, gnd, out)
    ("Regulator_Linear", "AMS1117-3.3", "Package_TO_SOT_SMD:SOT-223-3_TabPin2", 3.3, 1.3, ("VI", "GND", "VO")),
    ("Regulator_Linear", "LM7805_TO220", "Package_TO_SOT_THT:TO-220-3_Vertical", 5.0, 2.0, ("VI", "GND", "VO")),
    ("Regulator_Linear", "L7809", "Package_TO_SOT_THT:TO-220-3_Vertical", 9.0, 2.0, ("IN", "GND", "OUT")),
    ("Regulator_Linear", "L7812", "Package_TO_SOT_THT:TO-220-3_Vertical", 12.0, 2.0, ("IN", "GND", "OUT")),
]


def nearest(value, series=E24):
    """Nearest standard value in an E-series."""
    decade = 10 ** math.floor(math.log10(value))
    candidates = [s * decade for s in series] + [series[0] * decade * 10]
    return min(candidates, key=lambda c: abs(math.log(c / value)))


def fmt(value, unit=""):
    """Engineering notation for component values: 4700 -> '4.7k', 1e-7 -> '100n'."""
    for scale, prefix in ((1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "u"),
                          (1e-9, "n"), (1e-12, "p")):
        if abs(value) >= scale * 0.9999:
            text = f"{value / scale:.3g}"
            return f"{text}{prefix}{unit}"
    return f"{value:.3g}{unit}"


def human(value, unit):
    """Readable quantity for descriptions: 0.01 A -> '10 mA'."""
    for scale, prefix in ((1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "u")):
        if abs(value) >= scale * 0.9999:
            return f"{value / scale:.3g} {prefix}{unit}"
    return f"{value:.3g} {unit}"


OPENERS = ["", "Design ", "I need ", "Please make ", "Build "]


def phrase(rng, text):
    """Light variation in how a brief starts, so descriptions are not all one template."""
    opener = rng.choice(OPENERS)
    if not opener:
        return text
    return opener + text[0].lower() + text[1:]


def header(n):
    return (f'Part("Connector", "Conn_01x{n:02d}_Pin", '
            f'footprint="Connector_PinHeader_2.54mm:PinHeader_1x{n:02d}_P2.54mm_Vertical")')


PREAMBLE = '''R = lambda value, **kw: Part("Device", "R", value=value, footprint="{r}", **kw)
C = lambda value, **kw: Part("Device", "C", value=value, footprint="{c}", **kw)
'''.format(r=R_FP, c=C_FP)


def req(measure, low, high, name, **kw):
    parts = [repr(measure), f"{low:.6g}", f"{high:.6g}"]
    parts += [f"{k}={v!r}" for k, v in kw.items()]
    parts.append(f"name={name!r}")
    return f"require({', '.join(parts)})"


# ----------------------------------------------------------------------------
# Families. Each takes a random.Random and returns (description, code).
# ----------------------------------------------------------------------------

def led_resistor(rng):
    vcc = rng.choice([3.3, 5, 6, 9, 12, 15, 24])
    colour = rng.choice([c for c, vf in LED_VF.items() if vf < vcc - 0.8])
    i = rng.choice([0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025])
    vf = LED_VF[colour]
    r = nearest((vcc - vf) / i)
    desc = phrase(rng, f"A small board that lights a {colour} LED at about {human(i, 'A')} from a "
            f"{vcc:g} V supply on a 2-pin header, with a current-limiting resistor. "
            f"Name the supply nets VCC and GND.")
    code = f'''"""{colour.capitalize()} LED at ~{human(i, 'A')} from {vcc:g} V.
R = (Vcc - Vf) / I = ({vcc:g} - {vf}) / {i:g} = {(vcc - vf) / i:.0f} ohm -> {fmt(r)}."""
vcc, gnd, a = Net("VCC"), Net("GND"), Net("LED_A")
vcc.drive = gnd.drive = POWER
{PREAMBLE}
j = {header(2)}
led = Part("Device", "LED", value="{colour}", footprint="{LED_FP}")
j[1, 2] += vcc, gnd
vcc & R("{fmt(r)}") & a
a += led["A"]
led["K"] += gnd
{req("i(VCC)", i * 0.8, i * 1.2, f"LED current ~{human(i, 'A')}", sources={"VCC": vcc})}
'''
    return desc, code


def voltage_divider(rng):
    vin = rng.choice([3.3, 5, 9, 12, 15, 24])
    ratio = rng.uniform(0.1, 0.8)
    vout = round(vin * ratio, 2)
    total = rng.choice([10e3, 22e3, 47e3, 100e3])
    r2 = nearest(total * ratio)
    r1 = nearest(total - r2)
    actual = vin * r2 / (r1 + r2)
    desc = phrase(rng, f"Resistive divider that scales a {vin:g} V input down to about {vout:g} V for an "
            f"ADC input, total resistance around {fmt(total)}ohm, input and output on pin "
            f"headers. Name the nets VIN, OUT and GND.")
    code = f'''"""Divider: Vout = Vin * R2 / (R1 + R2) = {vin:g} * {fmt(r2)} / ({fmt(r1)} + {fmt(r2)}) = {actual:.3g} V."""
vin, out, gnd = Net("VIN"), Net("OUT"), Net("GND")
vin.drive = gnd.drive = POWER
{PREAMBLE}
j_in = {header(2)}
j_out = {header(2)}
j_in[1, 2] += vin, gnd
j_out[1, 2] += out, gnd
vin & R("{fmt(r1)}", tolerance=0.01) & out & R("{fmt(r2)}", tolerance=0.01) & gnd
{req("v(OUT)", vout * 0.93, vout * 1.07, f"~{vout:g} V out", sources={"VIN": vin})}
'''
    return desc, code


def rc_filter(rng):
    kind = rng.choice(["low", "high"])
    fc = nearest(10 ** rng.uniform(1, 5), E24)
    c = nearest(1 / (2 * math.pi * fc * rng.uniform(3e3, 30e3)), E6)
    r = nearest(1 / (2 * math.pi * fc * c))
    actual = 1 / (2 * math.pi * r * c)
    desc = phrase(rng, f"Passive first-order RC {kind}-pass filter with a {human(fc, 'Hz')} cutoff, input "
            f"and output on 2-pin headers. Name the nets IN, OUT and GND.")
    if kind == "low":
        wiring = 'inp & R("{r}") & out & C("{c}") & gnd'
        far = fc * 10
    else:
        wiring = 'inp & C("{c}") & out & R("{r}") & gnd'
        far = fc / 10
    wiring = wiring.format(r=fmt(r), c=fmt(c))
    code = f'''"""First-order RC {kind}-pass: fc = 1 / (2 pi R C) = 1 / (2 pi * {fmt(r)} * {fmt(c)}F) = {actual:.4g} Hz."""
inp, out, gnd = Net("IN"), Net("OUT"), Net("GND")
gnd.drive = POWER
{PREAMBLE}
j_in = {header(2)}
j_out = {header(2)}
j_in[1, 2] += inp, gnd
j_out[1, 2] += out, gnd
{wiring}
{req("gain(OUT,IN)", 0.6, 0.8, f"-3 dB at {human(fc, 'Hz')}", sources={"IN": "ac"}, analysis="ac", freq=fc)}
{req("gain(OUT,IN)", 0, 0.15, f"attenuates {human(far, 'Hz')}", sources={"IN": "ac"}, analysis="ac", freq=far)}
'''
    return desc, code


def linear_regulator(rng):
    lib, name, fp, vout, drop, (pin_in, pin_gnd, pin_out) = rng.choice(REGULATORS)
    vin = rng.choice([v for v in (5, 6, 7.5, 9, 10, 12, 13.8, 15, 18, 20, 24) if v >= vout + drop + 0.5])
    iload = rng.choice([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5])
    rload = vout / iload
    desc = phrase(rng, f"{vin:g} V to {vout:g} V linear regulator board for up to {human(iload, 'A')}, "
            f"with input and output 2-pin headers and the regulator's input and output "
            f"capacitors. Name the nets VIN, VOUT and GND.")
    code = f'''"""{vin:g} V -> {vout:g} V with a fixed {name}: needs Vin >= Vout + ~{drop:g} V dropout.
Load {human(iload, 'A')} = {rload:.3g} ohm; dissipation (Vin - Vout) * I = {(vin - vout) * iload:.2g} W."""
vin, vout, gnd = Net("VIN"), Net("VOUT"), Net("GND")
vin.drive = gnd.drive = POWER
{PREAMBLE}
j_in = {header(2)}
j_out = {header(2)}
reg = Part("{lib}", "{name}", footprint="{fp}")
j_in[1, 2] += vin, gnd
j_out[1, 2] += vout, gnd
reg["{pin_in}"] += vin
reg["{pin_gnd}"] += gnd
reg["{pin_out}"] += vout
vin & C("10u") & gnd
vout & C("22u") & gnd
{req("v(VOUT)", vout * 0.95, vout * 1.05, f"{vout:g} V at {human(iload, 'A')}", sources={"VIN": vin}, loads={"VOUT": round(rload, 3)})}
{req("v(VOUT)", vout * 0.95, vout * 1.05, f"{vout:g} V unloaded", sources={"VIN": vin})}
'''
    return desc, code


def _opamp(rng):
    sym, fp = rng.choice(OPAMPS)
    return sym, fp


def noninverting_amp(rng):
    sym, fp = _opamp(rng)
    gain = rng.choice([2, 3, 4, 5, 6, 8, 10, 11, 15, 20, 21, 25, 30, 33, 40, 50])
    rg = rng.choice([1e3, 2.2e3, 4.7e3, 10e3])
    rf = nearest(rg * (gain - 1))
    actual = 1 + rf / rg
    vcc = rng.choice([9, 12, 15]) if sym == "TL072" else rng.choice([5, 9, 12])
    load = rng.choice([2200, 4700, 10000, 47000])
    ftest = rng.choice([200, 500, 1000, 2000])
    desc = phrase(rng, f"Single-supply {vcc:g} V non-inverting op-amp amplifier with a gain of {gain} "
            f"at {human(ftest, 'Hz')}, AC-coupled input and output, biased at mid-supply, able to "
            f"drive a {fmt(load)}ohm load. Name the nets VCC, GND, IN and OUT.")
    code = f'''"""Non-inverting amplifier on {vcc:g} V: gain = 1 + Rf/Rg = 1 + {fmt(rf)}/{fmt(rg)} = {actual:.3g}.
Mid-supply reference from a 10k/10k divider; Rg returns to the reference through
a capacitor so the DC gain is 1 and the output sits at mid-supply."""
vcc, gnd, inp, out = Net("VCC"), Net("GND"), Net("IN"), Net("OUT")
vref, ni, fb, fbc, oa = Net("VREF"), Net("NI"), Net("FB"), Net("FBC"), Net("OA")
vcc.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(2)}
j_in = {header(2)}
j_out = {header(2)}
u = Part("Amplifier_Operational", "{sym}", footprint="{fp}")
j_pwr[1, 2] += vcc, gnd
j_in[1, 2] += inp, gnd
j_out[1, 2] += out, gnd
u["V+"] += vcc
u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("10k") & vref & R("10k") & gnd
vref & C("10u") & gnd
inp & C("1u") & ni & R("100k") & vref
oa & R("{fmt(rf)}", tolerance=0.01) & fb & R("{fmt(rg)}", tolerance=0.01) & fbc & C("100u") & vref
u[3] += ni
u[2] += fb
u[1] += oa
oa & C("10u") & out
u[5] += vref        # unused section as a follower at the reference
u[6] += u[7]
{req("gain(OUT,IN)", gain * 0.85, gain * 1.15, f"gain ~{gain} into {fmt(load)}", sources={"VCC": vcc, "IN": "ac"}, loads={"OUT": load}, analysis="ac", freq=float(ftest))}
'''
    return desc, code


def inverting_amp(rng):
    sym, fp = _opamp(rng)
    gain = rng.choice([1, 2, 3, 4.7, 5, 6.8, 10, 15, 22, 33, 47])
    rin = rng.choice([4.7e3, 10e3, 22e3])
    rf = nearest(rin * gain)
    actual = rf / rin
    vcc = rng.choice([9, 12, 15]) if sym == "TL072" else rng.choice([5, 9, 12])
    load = rng.choice([2200, 4700, 10000, 47000])
    ftest = rng.choice([200, 500, 1000, 2000])
    desc = phrase(rng, f"Single-supply {vcc:g} V inverting amplifier with a gain of -{gain:g} at "
            f"{human(ftest, 'Hz')}, AC-coupled input and output, biased at mid-supply, able to "
            f"drive a {fmt(load)}ohm load. Name the nets VCC, GND, IN and OUT.")
    code = f'''"""Inverting amplifier on {vcc:g} V: |gain| = Rf/Rin = {fmt(rf)}/{fmt(rin)} = {actual:.3g}."""
vcc, gnd, inp, out = Net("VCC"), Net("GND"), Net("IN"), Net("OUT")
vref, cin, inv, oa = Net("VREF"), Net("CIN"), Net("INV"), Net("OA")
vcc.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(2)}
j_in = {header(2)}
j_out = {header(2)}
u = Part("Amplifier_Operational", "{sym}", footprint="{fp}")
j_pwr[1, 2] += vcc, gnd
j_in[1, 2] += inp, gnd
j_out[1, 2] += out, gnd
u["V+"] += vcc
u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("10k") & vref & R("10k") & gnd
vref & C("10u") & gnd
inp & C("10u") & cin & R("{fmt(rin)}", tolerance=0.01) & inv
inv & R("{fmt(rf)}", tolerance=0.01) & oa
u[2] += inv
u[3] += vref
u[1] += oa
oa & C("10u") & out
u[5] += vref
u[6] += u[7]
{req("gain(OUT,IN)", gain * 0.85, gain * 1.15, f"|gain| ~{gain:g} into {fmt(load)}", sources={"VCC": vcc, "IN": "ac"}, loads={"OUT": load}, analysis="ac", freq=float(ftest))}
'''
    return desc, code


def sallen_key_lowpass(rng):
    sym, fp = rng.choice([("TL072", SOIC8), ("LM358", SOIC8)])
    fc = nearest(10 ** rng.uniform(1.7, 4.3), E24)
    c2 = nearest(1 / (2 * math.pi * fc * 10e3 * math.sqrt(2)), E12)
    c1 = nearest(2 * c2, E12)
    r = nearest(1 / (2 * math.pi * fc * math.sqrt(c1 * c2)))
    actual = 1 / (2 * math.pi * r * math.sqrt(c1 * c2))
    q = math.sqrt(c1 * c2) / (2 * c2)
    desc = phrase(rng, f"Second-order Butterworth low-pass filter (Sallen-Key, unity gain) with a "
            f"{human(fc, 'Hz')} cutoff, using an op-amp on +/-12 V supplies. Input, output and "
            f"power on pin headers. Name the nets IN, OUT, VPOS (+12 V), VNEG (-12 V) and GND.")
    code = f'''"""Unity-gain Sallen-Key low-pass: fc = 1/(2 pi R sqrt(C1 C2)) = {actual:.4g} Hz,
Q = sqrt(C1 C2)/(2 C2) = {q:.2f} (Butterworth 0.707), with R = {fmt(r)}, C1 = {fmt(c1)}F, C2 = {fmt(c2)}F."""
vpos, vneg, gnd = Net("VPOS"), Net("VNEG"), Net("GND")
inp, out, a, b = Net("IN"), Net("OUT"), Net("A"), Net("B")
vpos.drive = vneg.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(3)}
j_in = {header(2)}
j_out = {header(2)}
u = Part("Amplifier_Operational", "{sym}", footprint="{fp}")
j_pwr[1, 2, 3] += vpos, gnd, vneg
j_in[1, 2] += inp, gnd
j_out[1, 2] += out, gnd
u["V+"] += vpos
u["V-"] += vneg
vpos & C("100n") & gnd
vneg & C("100n") & gnd
inp & R("{fmt(r)}", tolerance=0.01) & a & R("{fmt(r)}", tolerance=0.01) & b
a & C("{fmt(c1)}", tolerance=0.05) & out
b & C("{fmt(c2)}", tolerance=0.05) & gnd
u[3] += b
u[2] += out
u[1] += out
u[5] += gnd
u[6] += u[7]
{req("gain(OUT,IN)", 0.9, 1.1, "passband gain ~1", sources={"VPOS": 12, "VNEG": -12, "IN": "ac"}, analysis="ac", freq=round(fc / 10, 3))}
{req("gain(OUT,IN)", 0.55, 0.85, f"-3 dB near {human(fc, 'Hz')}", sources={"VPOS": 12, "VNEG": -12, "IN": "ac"}, analysis="ac", freq=fc)}
{req("gain(OUT,IN)", 0, 0.025, "-40 dB/decade above cutoff", sources={"VPOS": 12, "VNEG": -12, "IN": "ac"}, analysis="ac", freq=fc * 10)}
'''
    return desc, code


def astable_555(rng):
    f = nearest(10 ** rng.uniform(-0.3, 3.5), E24)
    vcc = rng.choice([5, 6, 9, 12, 15])
    c = nearest(rng.uniform(1, 30) / (f * 1e5), E6)        # aims Ra + 2 Rb in ~10k-300k
    total = 1.44 / (f * c)
    ra = nearest(total * 0.1)
    rb = nearest((total - ra) / 2)
    actual = 1.44 / ((ra + 2 * rb) * c)
    duration = max(6 / actual, 6 * 10e-6)
    colour = rng.choice(["red", "green", "yellow"])
    r_led = nearest((vcc - 1.5 - LED_VF[colour]) / 0.01)
    desc = phrase(rng, f"555 timer astable oscillator running at about {human(f, 'Hz')} from a {vcc:g} V "
            f"supply, with a {colour} LED showing the output. Power on a 2-pin header. Name the "
            f"supply nets VCC and GND, and the 555 output net OUT.")
    code = f'''"""555 astable: f = 1.44 / ((Ra + 2 Rb) C) = 1.44 / (({fmt(ra)} + 2 * {fmt(rb)}) * {fmt(c)}F) = {actual:.4g} Hz.
LED resistor: (Vcc - 1.5 V output drop - Vf) / 10 mA -> {fmt(r_led)}."""
vcc, gnd, out = Net("VCC"), Net("GND"), Net("OUT")
dis, thr, cv, led_a = Net("DIS"), Net("THR"), Net("CV"), Net("LED_A")
vcc.drive = gnd.drive = POWER
{PREAMBLE}
j = {header(2)}
t = Part("Timer", "NE555D", footprint="{SOIC8}")
led = Part("Device", "LED", value="{colour}", footprint="{LED_FP}")
j[1, 2] += vcc, gnd
t["VCC"] += vcc
t["R"] += vcc
t["GND"] += gnd
t["DIS"] += dis
t["THR"] += thr
t["TR"] += thr
t["CV"] += cv
t["Q"] += out
vcc & R("{fmt(ra)}") & dis & R("{fmt(rb)}") & thr
thr & C("{fmt(c)}") & gnd
cv & C("10n") & gnd
vcc & C("100n") & gnd
out & R("{fmt(r_led)}") & led_a
led_a += led["A"]
led["K"] += gnd
{req("freq(OUT)", f * 0.75, f * 1.3, f"~{human(f, 'Hz')}", sources={"VCC": vcc}, analysis="tran", duration=round(duration, 6))}
'''
    return desc, code


ZENERS = [3.3, 3.9, 4.7, 5.1, 5.6, 6.2, 6.8, 8.2, 9.1]


def zener_regulator(rng):
    vz = rng.choice(ZENERS)
    vin = rng.choice([v for v in (9, 10, 12, 13.8, 15, 18, 24) if v > vz + 3])
    iload = rng.choice([0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015])
    rs = nearest((vin - vz) / (iload + 0.005), E12)
    desc = phrase(rng, f"Zener shunt regulator that makes {vz:g} V from a {vin:g} V input and can supply "
            f"up to {human(iload, 'A')}. Input and output on 2-pin headers. Name the nets VIN, OUT "
            f"and GND.")
    code = f'''"""Zener shunt regulator: Rs = (Vin - Vz) / (Iload + ~5 mA zener) = ({vin:g} - {vz:g}) / {iload + 0.005:g} -> {fmt(rs)}."""
vin, out, gnd = Net("VIN"), Net("OUT"), Net("GND")
vin.drive = gnd.drive = POWER
{PREAMBLE}
j_in = {header(2)}
j_out = {header(2)}
z = Part("Device", "D_Zener", value="{vz:g}V", footprint="Diode_SMD:D_SOD-123")
j_in[1, 2] += vin, gnd
j_out[1, 2] += out, gnd
vin & R("{fmt(rs)}") & out
z["K"] += out
z["A"] += gnd
out & C("10u") & gnd
{req("v(OUT)", vz * 0.92, vz * 1.08, f"{vz:g} V at {human(iload, 'A')}", sources={"VIN": vin}, loads={"OUT": round(vz / iload, 1)})}
{req("v(OUT)", vz * 0.92, vz * 1.1, f"{vz:g} V unloaded", sources={"VIN": vin})}
'''
    return desc, code


def mosfet_switch(rng):
    vload = rng.choice([5, 9, 12, 15, 24])
    vlogic = rng.choice([3.3, 5])
    iload = rng.choice([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0])
    rload = vload / iload
    sym = "AO3400A" if iload <= 0.5 else "IRLZ44N"
    fp = "Package_TO_SOT_SMD:SOT-23" if sym == "AO3400A" else "Package_TO_SOT_THT:TO-220-3_Vertical"
    desc = phrase(rng, f"Low-side N-channel MOSFET switch that lets a {vlogic:g} V logic signal switch a "
            f"{vload:g} V, {human(iload, 'A')} load, with a gate resistor, a gate pull-down so the "
            f"load stays off when the input floats, and a flyback diode. The load connects between "
            f"V{vload:g} and LOAD_NEG. Name the nets V{vload:g}, GND, LOAD_NEG and CTRL.")
    vnet = f"V{vload:g}"
    code = f'''"""Low-side switch: {sym} (logic-level N-MOSFET), 100 R gate resistor, 10k pull-down,
flyback diode across the load (cathode to {vnet})."""
vsup, gnd, load_neg, ctrl, gate = Net("{vnet}"), Net("GND"), Net("LOAD_NEG"), Net("CTRL"), Net("GATE")
vsup.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(2)}
j_load = {header(2)}
j_ctrl = {header(2)}
q = Part("Transistor_FET", "{sym}", footprint="{fp}")
d = Part("Diode", "1N4148W", footprint="Diode_SMD:D_SOD-123")
j_pwr[1, 2] += vsup, gnd
j_load[1, 2] += vsup, load_neg
j_ctrl[1, 2] += ctrl, gnd
q["D"] += load_neg
q["S"] += gnd
q["G"] += gate
ctrl & R("100") & gate
gate & R("10k") & gnd
d["K"] += vsup
d["A"] += load_neg
vsup & C("10u") & gnd
{req("v(LOAD_NEG)", 0, vload * 0.05, "load on", sources={vnet: vload, "CTRL": vlogic}, loads={"LOAD_NEG": {"to": vnet, "r": round(rload, 3)}})}
{req("v(LOAD_NEG)", vload * 0.95, vload * 1.01, "load off", sources={vnet: vload, "CTRL": 0}, loads={"LOAD_NEG": {"to": vnet, "r": round(rload, 3)}})}
{req("v(LOAD_NEG)", vload * 0.95, vload * 1.01, "off with CTRL floating", sources={vnet: vload}, loads={"LOAD_NEG": {"to": vnet, "r": round(rload, 3)}})}
'''
    return desc, code


def bjt_led_switch(rng):
    vcc = rng.choice([5, 6, 9, 12, 15])
    vlogic = rng.choice([1.8, 3.3, 5])
    colour = rng.choice(list(LED_VF))
    i = rng.choice([0.005, 0.01, 0.015, 0.02, 0.03])
    vf = LED_VF[colour]
    r_led = nearest((vcc - vf - 0.1) / i)
    ib = i / 10                     # hard saturation: beta_forced = 10
    rb = nearest((vlogic - 0.7) / ib)
    desc = phrase(rng, f"NPN transistor switch that lets a {vlogic:g} V logic input turn on a {colour} LED "
            f"at about {human(i, 'A')} from a {vcc:g} V supply, with a base resistor and a "
            f"pull-down so the LED is off when the input floats. Name the nets VCC, GND and CTRL.")
    code = f'''"""NPN low-side LED switch: R_led = (Vcc - Vf - Vce_sat) / I = ({vcc:g} - {vf} - 0.1) / {i:g} -> {fmt(r_led)};
base current I/10 for saturation: Rb = ({vlogic:g} - 0.7) / {ib:g} -> {fmt(rb)}."""
vcc, gnd, ctrl, base, col, a = Net("VCC"), Net("GND"), Net("CTRL"), Net("BASE"), Net("COL"), Net("LED_A")
vcc.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(2)}
j_ctrl = {header(2)}
q = Part("Transistor_BJT", "MMBT3904", footprint="Package_TO_SOT_SMD:SOT-23")
led = Part("Device", "LED", value="{colour}", footprint="{LED_FP}")
j_pwr[1, 2] += vcc, gnd
j_ctrl[1, 2] += ctrl, gnd
vcc & R("{fmt(r_led)}") & a
a += led["A"]
led["K"] += col
q["C"] += col
q["E"] += gnd
q["B"] += base
ctrl & R("{fmt(rb)}") & base
base & R("100k") & gnd
vcc & C("100n") & gnd
{req("i(VCC)", i * 0.75, i * 1.25, f"LED on at ~{human(i, 'A')}", sources={"VCC": vcc, "CTRL": vlogic})}
{req("i(VCC)", 0, i * 0.02, "LED off with CTRL low", sources={"VCC": vcc, "CTRL": 0})}
'''
    return desc, code


def buffered_reference(rng):
    sym, fp = rng.choice([("LM358", SOIC8), ("TL072", SOIC8)])
    vcc = rng.choice([9, 12, 15]) if sym == "TL072" else rng.choice([5, 9, 12])
    vref = round(vcc * rng.uniform(0.3, 0.6), 2)
    r2 = nearest(10e3 * vref / vcc * 2)
    r1 = nearest(r2 * (vcc - vref) / vref)
    actual = vcc * r2 / (r1 + r2)
    vref = round(actual, 2)            # describe what the standard values actually give
    rl = rng.choice([470, 1000, 2200, 4700, 10000])
    desc = phrase(rng, f"A {vref:g} V reference made from a resistor divider on a {vcc:g} V supply, buffered "
            f"by an op-amp voltage follower so it can drive a {fmt(rl)}ohm load. Power and the "
            f"reference output on pin headers. Name the nets VCC, GND and VREF_OUT.")
    code = f'''"""Divider {fmt(r1)}/{fmt(r2)} gives {actual:.3g} V; an op-amp follower drives the load."""
vcc, gnd, out, div = Net("VCC"), Net("GND"), Net("VREF_OUT"), Net("DIV")
vcc.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(2)}
j_out = {header(2)}
u = Part("Amplifier_Operational", "{sym}", footprint="{fp}")
j_pwr[1, 2] += vcc, gnd
j_out[1, 2] += out, gnd
u["V+"] += vcc
u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("{fmt(r1)}", tolerance=0.01) & div & R("{fmt(r2)}", tolerance=0.01) & gnd
div & C("100n") & gnd
u[3] += div
u[2] += out
u[1] += out
u[5] += div
u[6] += u[7]
{req("v(VREF_OUT)", vref * 0.95, vref * 1.05, f"{vref:g} V into {fmt(rl)}", sources={"VCC": vcc}, loads={"VREF_OUT": rl})}
'''
    return desc, code


def _supplies_pm(rng):
    v = rng.choice([12, 15])
    return v, {"VPOS": v, "VNEG": -v}


def instrumentation_amp(rng):
    gain = rng.choice([10, 20, 30, 50, 200, 500])      # the benchmark board is gain 100
    v, sup = _supplies_pm(rng)
    diff_gain = 1 if gain <= 20 else 10
    g1 = gain / diff_gain
    rf = 10e3
    rg = nearest(2 * rf / (g1 - 1))
    actual = (1 + 2 * rf / rg) * diff_gain
    rd = 10e3
    rd2 = rd * diff_gain
    desc = phrase(rng, f"Three-op-amp instrumentation amplifier with a differential gain of about "
            f"{gain}, set by one gain resistor, with buffered high-impedance inputs and good "
            f"common-mode rejection, on +/-{v} V supplies. Inputs, output and power on pin headers. "
            f"Name the nets VPOS, VNEG, GND, IN_P, IN_N and OUT.")
    code = f'''"""Instrumentation amplifier on +/-{v} V: gain = (1 + 2*{fmt(rf)}/{fmt(rg)}) * {fmt(rd2)}/{fmt(rd)} = {actual:.4g}.
Input buffers for high impedance; 0.1% resistors in the difference stage for common-mode rejection."""
vpos, vneg, gnd = Net("VPOS"), Net("VNEG"), Net("GND")
inp, inn, out = Net("IN_P"), Net("IN_N"), Net("OUT")
a1, a2, g1, g2, ni, inv = Net("A1"), Net("A2"), Net("G1"), Net("G2"), Net("NI"), Net("INV")
vpos.drive = vneg.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(3)}
j_in = {header(3)}
j_out = {header(2)}
j_pwr[1, 2, 3] += vpos, gnd, vneg
j_in[1, 2, 3] += inp, inn, gnd
j_out[1, 2] += out, gnd
u1 = Part("Amplifier_Operational", "TL072", footprint="{SOIC8}")
u2 = Part("Amplifier_Operational", "TL072", footprint="{SOIC8}")
for u in (u1, u2):
    u["V+"] += vpos
    u["V-"] += vneg
vpos & C("100n") & gnd
vneg & C("100n") & gnd
u1[3] += inp
u1[2] += g1
u1[1] += a1
u1[5] += inn
u1[6] += g2
u1[7] += a2
a1 & R("{fmt(rf)}", tolerance=0.01) & g1 & R("{fmt(rg)}", tolerance=0.01) & g2 & R("{fmt(rf)}", tolerance=0.01) & a2
a2 & R("{fmt(rd)}", tolerance=0.001) & inv & R("{fmt(rd2)}", tolerance=0.001) & out
a1 & R("{fmt(rd)}", tolerance=0.001) & ni & R("{fmt(rd2)}", tolerance=0.001) & gnd
u2[2] += inv
u2[3] += ni
u2[1] += out
u2[5] += gnd        # unused section as a follower at GND
u2[6] += u2[7]
{req("gain(OUT,IN_P)", actual * 0.88, actual * 1.12, f"differential gain ~{gain}", sources={**sup, "IN_P": "ac", "IN_N": 0}, analysis="ac", freq=100.0)}
{req("gain(OUT,IN_P)", 0, max(0.05, actual * 0.002), "common-mode rejection", sources={**sup, "IN_P": "ac", "IN_N": "ac"}, analysis="ac", freq=100.0)}
{req("i(IN_P)", 0, 1e-6, "high input impedance", sources={**sup, "IN_P": 0.1, "IN_N": 0.1})}
'''
    return desc, code


def mfb_bandpass(rng):
    while True:
        f0 = rng.choice([200, 300, 500, 700, 1500, 2000, 3000, 5000])   # benchmark: 1 kHz
        q = rng.choice([1, 1.5, 2, 3])
        h0 = rng.choice([1, 2])
        if 2 * q * q > h0:
            break
    v, sup = _supplies_pm(rng)
    w = 2 * math.pi * f0
    c = nearest(2 * q / (w * 47e3), E6)          # R3 near 47k
    r1 = nearest(q / (h0 * w * c))
    r2 = nearest(q / ((2 * q * q - h0) * w * c))
    r3 = nearest(2 * q / (w * c))
    desc = phrase(rng, f"Multiple-feedback active band-pass filter centred at {human(f0, 'Hz')} with a Q "
            f"of about {q:g} and a gain of about {h0} at the centre frequency, using an op-amp on "
            f"+/-{v} V supplies. Input, output and power on pin headers. Name the nets VPOS, VNEG, "
            f"GND, IN and OUT.")
    off = lambda f: h0 / math.sqrt(1 + q * q * (f / f0 - f0 / f) ** 2)
    code = f'''"""MFB band-pass: f0 = {human(f0, 'Hz')}, Q = {q:g}, centre gain {h0}, C = {fmt(c)}:
R1 = Q/(H0 w C) -> {fmt(r1)}, R2 = Q/((2Q^2 - H0) w C) -> {fmt(r2)}, R3 = 2Q/(w C) -> {fmt(r3)}."""
vpos, vneg, gnd, inp, out = Net("VPOS"), Net("VNEG"), Net("GND"), Net("IN"), Net("OUT")
x, inv = Net("X"), Net("INV")
vpos.drive = vneg.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(3)}
j_in = {header(2)}
j_out = {header(2)}
j_pwr[1, 2, 3] += vpos, gnd, vneg
j_in[1, 2] += inp, gnd
j_out[1, 2] += out, gnd
u = Part("Amplifier_Operational", "TL072", footprint="{SOIC8}")
u["V+"] += vpos
u["V-"] += vneg
vpos & C("100n") & gnd
vneg & C("100n") & gnd
inp & R("{fmt(r1)}", tolerance=0.01) & x
x & R("{fmt(r2)}", tolerance=0.01) & gnd
x & C("{fmt(c)}", tolerance=0.05) & out
x & C("{fmt(c)}", tolerance=0.05) & inv
inv & R("{fmt(r3)}", tolerance=0.01) & out
u[2] += inv
u[3] += gnd
u[1] += out
u[5] += gnd         # unused section as a follower at GND
u[6] += u[7]
{req("gain(OUT,IN)", h0 * 0.75, h0 * 1.25, f"gain ~{h0} at {human(f0, 'Hz')}", sources={**sup, "IN": "ac"}, analysis="ac", freq=float(f0))}
{req("gain(OUT,IN)", 0, off(f0 / 5) * 1.6 + 0.02, f"rejects {human(f0 / 5, 'Hz')}", sources={**sup, "IN": "ac"}, analysis="ac", freq=float(f0 / 5))}
{req("gain(OUT,IN)", 0, off(f0 * 5) * 1.6 + 0.02, f"rejects {human(f0 * 5, 'Hz')}", sources={**sup, "IN": "ac"}, analysis="ac", freq=float(f0 * 5))}
'''
    return desc, code


def window_comparator(rng):
    while True:
        vcc = rng.choice([5, 9, 12])
        lo = rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 4]) * (vcc / 5)
        hi = lo + rng.choice([0.5, 1, 1.5, 2]) * (vcc / 5)
        if hi < vcc - 1 and (vcc, lo, hi) != (5, 2, 3):    # 2-3 V on 5 V is the benchmark
            break
    rtot = 50e3
    rb = nearest(rtot * lo / vcc)
    rm = nearest(rtot * (hi - lo) / vcc)
    rt = nearest(rtot * (vcc - hi) / vcc)
    s = rb + rm + rt
    alo, ahi = vcc * rb / s, vcc * (rb + rm) / s
    margin = min(0.5, (ahi - alo) / 4)
    inside, below, above = (alo + ahi) / 2, alo - margin, ahi + margin
    desc = phrase(rng, f"Window comparator on a {vcc:g} V supply: the output OUT is high while the "
            f"input IN is between {lo:.3g} V and {hi:.3g} V and low otherwise. Use a dual op-amp as two "
            f"comparators with thresholds from a resistor divider. Name the nets VCC, GND, IN and OUT.")
    code = f'''"""Window comparator on {vcc:g} V: ladder {fmt(rt)}/{fmt(rm)}/{fmt(rb)} sets {ahi:.3g} V and {alo:.3g} V.
U1A is high while IN is below the upper threshold, U1B while it is above the lower one;
diodes to each output pull OUT low unless both are high."""
vcc, gnd, inp, out = Net("VCC"), Net("GND"), Net("IN"), Net("OUT")
vh, vl, oa, ob = Net("VH"), Net("VL"), Net("OA"), Net("OB")
vcc.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(2)}
j_in = {header(2)}
j_out = {header(2)}
j_pwr[1, 2] += vcc, gnd
j_in[1, 2] += inp, gnd
j_out[1, 2] += out, gnd
u = Part("Amplifier_Operational", "LM358", footprint="{SOIC8}")
u["V+"] += vcc
u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("{fmt(rt)}", tolerance=0.01) & vh & R("{fmt(rm)}", tolerance=0.01) & vl & R("{fmt(rb)}", tolerance=0.01) & gnd
u[3] += vh
u[2] += inp
u[1] += oa
u[5] += inp
u[6] += vl
u[7] += ob
vcc & R("10k") & out
d1 = Part("Diode", "1N4148W", footprint="Diode_SMD:D_SOD-123")
d2 = Part("Diode", "1N4148W", footprint="Diode_SMD:D_SOD-123")
d1["A"] += out
d1["K"] += oa
d2["A"] += out
d2["K"] += ob
{req("v(OUT)", vcc * 0.9, vcc + 0.1, f"high inside the window ({inside:.3g} V)", sources={"VCC": vcc, "IN": round(inside, 3)})}
{req("v(OUT)", 0, 1, f"low below the window ({below:.3g} V)", sources={"VCC": vcc, "IN": round(below, 3)})}
{req("v(OUT)", 0, 1, f"low above the window ({above:.3g} V)", sources={"VCC": vcc, "IN": round(above, 3)})}
'''
    return desc, code


def two_stage_amp(rng):
    total = rng.choice([100, 200, 300, 400, 600, 2000])      # benchmark: 1000
    vcc = rng.choice([9, 12, 15])
    stage = math.sqrt(total)
    rg = 1e3
    rf = nearest(rg * (stage - 1))
    actual = (1 + rf / rg) ** 2
    desc = phrase(rng, f"Two-stage audio amplifier with a total gain of about {total} from 100 Hz to "
            f"10 kHz, from a single {vcc:g} V supply: two op-amp gain stages with a mid-supply "
            f"reference, AC coupling at the input, between the stages and at the output, able to "
            f"drive a 10k load. Name the nets VCC, GND, IN and OUT.")
    tests = "\n".join(req("gain(OUT,IN)", actual * 0.7, actual * 1.3, f"gain ~{total} at {human(f, 'Hz')}",
                          sources={"VCC": vcc, "IN": "ac"}, loads={"OUT": 10000}, analysis="ac",
                          freq=float(f)) for f in (100, 1000, 10000))
    code = f'''"""Two non-inverting stages of gain 1 + {fmt(rf)}/{fmt(rg)} = {1 + rf / rg:.3g} each ({actual:.4g} total)
on {vcc:g} V, mid-supply reference, AC coupling at the input, between stages and at the output."""
vcc, gnd, inp, out = Net("VCC"), Net("GND"), Net("IN"), Net("OUT")
ref, n1, i1, f1, o1 = Net("REF"), Net("N1"), Net("I1"), Net("F1"), Net("O1")
n2, i2, f2, o2 = Net("N2"), Net("I2"), Net("F2"), Net("O2")
vcc.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(2)}
j_in = {header(2)}
j_out = {header(2)}
j_pwr[1, 2] += vcc, gnd
j_in[1, 2] += inp, gnd
j_out[1, 2] += out, gnd
u = Part("Amplifier_Operational", "TL072", footprint="{SOIC8}")
u["V+"] += vcc
u["V-"] += gnd
vcc & C("100n") & gnd
vcc & R("10k") & ref & R("10k") & gnd
ref & C("10u") & gnd
inp & C("1u") & n1 & R("100k") & ref
u[3] += n1
u[2] += i1
u[1] += o1
i1 & R("{fmt(rf)}", tolerance=0.01) & o1
i1 & R("{fmt(rg)}", tolerance=0.01) & f1 & C("10u") & gnd
o1 & C("1u") & n2 & R("100k") & ref
u[5] += n2
u[6] += i2
u[7] += o2
i2 & R("{fmt(rf)}", tolerance=0.01) & o2
i2 & R("{fmt(rg)}", tolerance=0.01) & f2 & C("10u") & gnd
o2 & C("10u") & out
{tests}
'''
    return desc, code


# Gate-level functions built from 74HC00 NANDs (XOR is a benchmark board).
NAND_FUNCTIONS = {
    "AND": (lambda a, b: a and b, ["N1 = A nand B", "OUT = N1 nand N1"],
            [(1, 2, 3, "a", "b", "n1"), (4, 5, 6, "n1", "n1", "out")]),
    "OR": (lambda a, b: a or b, ["N1 = A nand A", "N2 = B nand B", "OUT = N1 nand N2"],
           [(1, 2, 3, "a", "a", "n1"), (4, 5, 6, "b", "b", "n2"), (9, 10, 8, "n1", "n2", "out")]),
    "NOR": (lambda a, b: not (a or b), ["N1 = A nand A", "N2 = B nand B", "N3 = N1 nand N2",
                                        "OUT = N3 nand N3"],
            [(1, 2, 3, "a", "a", "n1"), (4, 5, 6, "b", "b", "n2"), (9, 10, 8, "n1", "n2", "n3"),
             (12, 13, 11, "n3", "n3", "out")]),
}


def nand_logic(rng):
    func = rng.choice(sorted(NAND_FUNCTIONS))
    vcc = rng.choice([3.3, 5])
    pull = rng.choice([4.7e3, 10e3, 47e3, 100e3])
    truth, steps, gates = NAND_FUNCTIONS[func]
    desc = phrase(rng, f"{func} function built only from the NAND gates of a 74HC00 on {vcc:g} V, with "
            f"{fmt(pull)} pull-down resistors on the inputs A and B so they read low when unconnected, "
            f"inputs and output on pin headers. Name the nets VCC, GND, A, B and OUT.")
    wiring = "\n".join(f"u[{i1}] += {x}\nu[{i2}] += {y}\nu[{o}] += {z}" for i1, i2, o, x, y, z in gates)
    used = {g[0] for g in gates}
    spare = [(1, 2, 3), (4, 5, 6), (9, 10, 8), (12, 13, 11)]
    unused = "\n".join(f"u[{i1}] += gnd\nu[{i2}] += gnd" for i1, i2, o in spare if i1 not in used)
    tests = "\n".join(
        req("v(OUT)", vcc * 0.9 if truth(a, b) else 0, vcc + 0.1 if truth(a, b) else vcc * 0.1,
            f"A={int(a)} B={int(b)} -> {int(bool(truth(a, b)))}",
            sources={"VCC": vcc, "A": vcc if a else 0, "B": vcc if b else 0})
        for a in (False, True) for b in (False, True))
    code = f'''"""{func} from 74HC00 NAND gates: {'; '.join(steps)}.
{fmt(pull)} pull-downs keep unconnected inputs low; unused gate inputs are tied to GND."""
vcc, gnd, a, b, out = Net("VCC"), Net("GND"), Net("A"), Net("B"), Net("OUT")
n1, n2, n3 = Net("N1"), Net("N2"), Net("N3")
vcc.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(2)}
j_in = {header(3)}
j_out = {header(2)}
j_pwr[1, 2] += vcc, gnd
j_in[1, 2, 3] += a, b, gnd
j_out[1, 2] += out, gnd
u = Part("74xx", "74HC00", footprint="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm")
u["VCC"] += vcc
u["GND"] += gnd
vcc & C("100n") & gnd
a & R("{fmt(pull)}") & gnd
b & R("{fmt(pull)}") & gnd
{wiring}
{unused}
{tests}
'''
    return desc, code


def current_mirror(rng):
    while True:
        vcc = rng.choice([5, 9, 12, 15, 24])
        i = rng.choice([0.0002, 0.0005, 0.001, 0.002, 0.005])
        if (vcc, i) != (12, 0.001):                       # the benchmark board
            break
    re_ = nearest(0.3 / i)                                # ~0.3 V of emitter degeneration
    rref = nearest((vcc - 0.65) / i - re_)
    actual = (vcc - 0.65) / (rref + re_)
    vlo, vhi = 1.5, vcc - 1.5
    desc = phrase(rng, f"NPN transistor current mirror that sinks about {human(i, 'A')} into OUT from a "
            f"{vcc:g} V supply, set by a single resistor, staying within 15% for OUT voltages from "
            f"{vlo:g} V to {vhi:g} V. Name the nets VCC, GND and OUT.")
    code = f'''"""NPN mirror: Iref = (Vcc - Vbe) / (Rref + Re) = ({vcc:g} - 0.65) / ({fmt(rref)} + {fmt(re_)}) = {human(actual, 'A')};
{fmt(re_)} emitter degeneration in both legs raises the output resistance."""
vcc, gnd, out, ref, e1, e2 = Net("VCC"), Net("GND"), Net("OUT"), Net("REF"), Net("E1"), Net("E2")
vcc.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(2)}
j_out = {header(2)}
j_pwr[1, 2] += vcc, gnd
j_out[1, 2] += out, gnd
q1 = Part("Transistor_BJT", "BC847", footprint="Package_TO_SOT_SMD:SOT-23")
q2 = Part("Transistor_BJT", "BC847", footprint="Package_TO_SOT_SMD:SOT-23")
vcc & R("{fmt(rref)}", tolerance=0.01) & ref
q1["C"] += ref
q1["B"] += ref
q1["E"] += e1
q2["C"] += out
q2["B"] += ref
q2["E"] += e2
e1 & R("{fmt(re_)}", tolerance=0.01) & gnd
e2 & R("{fmt(re_)}", tolerance=0.01) & gnd
vcc & C("100n") & gnd
{req("i(OUT)", i * 0.85, i * 1.15, f"~{human(i, 'A')} with OUT at {vlo:g} V", sources={"VCC": vcc, "OUT": vlo})}
{req("i(OUT)", i * 0.85, i * 1.15, f"~{human(i, 'A')} with OUT at {vhi:g} V", sources={"VCC": vcc, "OUT": vhi})}
'''
    return desc, code


def high_side_switch(rng):
    vsup = rng.choice([5, 9, 15, 24])                     # benchmark: 12 V
    vlogic = rng.choice([3.3, 5])
    iload = rng.choice([0.1, 0.25, 0.5, 1.0])
    rload = vsup / iload
    vnet = f"V{vsup:g}"
    # Gate held at Vsup by 10k; the NPN pulls it down through Rs, so |Vgs| = Vsup * 10k / (10k + Rs).
    rs = 0 if vsup <= 9 else nearest(10e3 * (vsup / 8 - 1))
    vgs = vsup if rs == 0 else vsup * 10e3 / (10e3 + rs)
    pull_down = f'gate & R("{fmt(rs)}") & col' if rs else "gate += col"
    through = f" through {fmt(rs)}" if rs else ""
    desc = phrase(rng, f"High-side switch that lets a {vlogic:g} V logic input CTRL connect a {vsup:g} V "
            f"supply to a {human(iload, 'A')} load through a P-channel MOSFET, driven by an NPN "
            f"transistor, keeping the gate-source voltage within the MOSFET's +/-12 V rating, with the "
            f"load off when CTRL is floating. The load connects between LOAD_POS and GND. Name the "
            f"nets {vnet}, GND, CTRL and LOAD_POS.")
    code = f'''"""High-side switch on {vsup:g} V: AO3401A P-MOSFET with its gate held at {vnet} by 10k; a BC847
pulls the gate down{through} so |Vgs| = {vgs:.3g} V. 10k base resistor, 100k base pull-down."""
vsup, gnd, ctrl, load = Net("{vnet}"), Net("GND"), Net("CTRL"), Net("LOAD_POS")
gate, col, base = Net("GATE"), Net("COL"), Net("BASE")
vsup.drive = gnd.drive = POWER
{PREAMBLE}
j_pwr = {header(2)}
j_load = {header(2)}
j_ctrl = {header(2)}
j_pwr[1, 2] += vsup, gnd
j_load[1, 2] += load, gnd
j_ctrl[1, 2] += ctrl, gnd
q = Part("Transistor_FET", "AO3401A", footprint="Package_TO_SOT_SMD:SOT-23")
q["S"] += vsup
q["D"] += load
q["G"] += gate
vsup & R("10k") & gate
{pull_down}
n = Part("Transistor_BJT", "BC847", footprint="Package_TO_SOT_SMD:SOT-23")
n["C"] += col
n["E"] += gnd
n["B"] += base
ctrl & R("10k") & base & R("100k") & gnd
vsup & C("10u") & gnd
{req("v(LOAD_POS)", vsup * 0.95, vsup + 0.01, f"load on at CTRL={vlogic:g} V", sources={vnet: vsup, "CTRL": vlogic}, loads={"LOAD_POS": rload})}
{req("v(LOAD_POS)", 0, 0.5, "load off at CTRL=0 V", sources={vnet: vsup, "CTRL": 0}, loads={"LOAD_POS": rload})}
{req("v(LOAD_POS)", 0, 0.5, "load off with CTRL floating", sources={vnet: vsup}, loads={"LOAD_POS": rload})}
'''
    return desc, code


def led_bar(rng):
    while True:
        vcc = rng.choice([5, 9, 12])
        n = rng.choice([2, 3, 4])
        step = rng.choice([0.5, 1, 1.5, 2]) * (vcc / 5)
        first = rng.choice([0.5, 1, 1.5]) * (vcc / 5)
        thr = [first + k * step for k in range(n)]
        if thr[-1] < vcc - 1 and not (vcc == 5 and n == 3 and thr == [1, 2, 3]):   # benchmark
            break
    colour = rng.choice(["red", "green", "yellow"])
    vf = LED_VF[colour]
    iled = rng.choice([0.005, 0.01])
    rled = nearest((vcc - vf) / iled)
    # Ladder from VCC: R_top to the highest threshold, equal rungs between thresholds, R_bot to GND.
    unit = 10e3
    r_bot = nearest(unit * first / step)
    r_top = nearest(unit * (vcc - thr[-1]) / step)
    total = r_top + unit * (n - 1) + r_bot
    actual = [vcc * (r_bot + k * unit) / total for k in range(n)]
    names = [f"LED{k + 1}" for k in range(n)]
    sections = [(3, 2, 1), (5, 6, 7), (10, 9, 8), (12, 13, 14)]
    chain = ['vcc & R("{}", tolerance=0.01)'.format(fmt(r_top))]
    chain += [f't{k} & R("{fmt(unit)}", tolerance=0.01)' for k in range(n, 1, -1)]
    chain += ['t1 & R("{}", tolerance=0.01) & gnd'.format(fmt(r_bot))]
    lines = [
        'vcc, gnd, inp = Net("VCC"), Net("GND"), Net("IN")',
        *(f't{k + 1} = Net("T{k + 1}")' for k in range(n)),
        *(f'led{k + 1}_a = Net("{names[k]}")' for k in range(n)),
        *(f'o{k + 1} = Net("O{k + 1}")' for k in range(n)),
        "vcc.drive = gnd.drive = POWER",
        PREAMBLE,
        f"j_pwr = {header(2)}",
        f"j_in = {header(2)}",
        "j_pwr[1, 2] += vcc, gnd",
        "j_in[1, 2] += inp, gnd",
        'u = Part("Amplifier_Operational", "LM324", footprint="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm")',
        'u["V+"] += vcc',
        'u["V-"] += gnd',
        'vcc & C("100n") & gnd',
        " & ".join(chain),
    ]
    for k in range(n):
        plus, minus, o = sections[k]
        lines += [f"u[{plus}] += inp", f"u[{minus}] += t{k + 1}", f"u[{o}] += o{k + 1}",
                  f'led{k + 1} = Part("Device", "LED", value="{colour}", footprint="{LED_FP}")',
                  f'o{k + 1} & R("{fmt(rled)}") & led{k + 1}_a',
                  f'led{k + 1}_a += led{k + 1}["A"]', f'led{k + 1}["K"] += gnd']
    for k in range(n, 4):
        plus, minus, o = sections[k]
        lines += [f"u[{plus}] += gnd        # unused section as a follower of GND",
                  f"u[{minus}] += u[{o}]"]
    for k in range(n):
        gap = min(0.4, step / 3)
        lines.append(req(f"v({names[k]})", vf - 0.4, vf + 0.5, f"{names[k]} lit at {actual[k] + gap:.3g} V",
                         sources={"VCC": vcc, "IN": round(actual[k] + gap, 3)}))
        lines.append(req(f"v({names[k]})", 0, 0.5, f"{names[k]} dark at {actual[k] - gap:.3g} V",
                         sources={"VCC": vcc, "IN": round(actual[k] - gap, 3)}))
    thr_text = ", ".join(f"{t:.3g} V" for t in thr)
    desc = phrase(rng, f"{n}-LED bar-graph voltage indicator on {vcc:g} V: {colour} LEDs light one after "
            f"another as the input IN rises past {thr_text}, using comparators from an LM324 quad "
            f"op-amp, a resistor ladder for the thresholds, and a current-limiting resistor for each "
            f"LED. Name the nets VCC, GND and IN, and the LED anode nets "
            + ", ".join(f"{nm} ({t:.3g} V)" for nm, t in zip(names, thr)) + ".")
    doc = (f'"""{n}-LED bar graph on {vcc:g} V: ladder sets {", ".join(f"{a:.3g} V" for a in actual)};\n'
           f"LM324 sections compare IN with each threshold and drive a {colour} LED through "
           f'{fmt(rled)} (~{human((vcc - vf) / rled, "A")})."""')
    return desc, doc + "\n" + "\n".join(lines) + "\n"


FAMILIES = {
    "led_resistor": led_resistor,
    "voltage_divider": voltage_divider,
    "rc_filter": rc_filter,
    "linear_regulator": linear_regulator,
    "noninverting_amp": noninverting_amp,
    "inverting_amp": inverting_amp,
    "sallen_key_lowpass": sallen_key_lowpass,
    "astable_555": astable_555,
    "zener_regulator": zener_regulator,
    "mosfet_switch": mosfet_switch,
    "bjt_led_switch": bjt_led_switch,
    "buffered_reference": buffered_reference,
    "instrumentation_amp": instrumentation_amp,
    "mfb_bandpass": mfb_bandpass,
    "window_comparator": window_comparator,
    "two_stage_amp": two_stage_amp,
    "nand_logic": nand_logic,
    "current_mirror": current_mirror,
    "high_side_switch": high_side_switch,
    "led_bar": led_bar,
}


def normalize(description):
    """A description without its varied opener, for comparing briefs."""
    d = description.strip()
    for opener in sorted(OPENERS, key=len, reverse=True):
        if opener and d.startswith(opener):
            d = d[len(opener):]
            break
    return re.sub(r"\s+", " ", d).lower().rstrip(".")


def _benchmark_briefs():
    import json
    from pathlib import Path
    path = Path(__file__).parent / "benchmarks" / "boards.json"
    return {normalize(c["description"]) for c in json.loads(path.read_text())}


def generate(n, seed=0, families=None, max_misses=50):
    """n boards, round-robin over families. Returns dicts with id, family, description, code.

    A family that keeps producing descriptions already seen is retired, so a
    small parameter space cannot stall the run.
    """
    rng = random.Random(seed)
    held_out = _benchmark_briefs()   # never train on a benchmark board
    active = list(families or FAMILIES)
    misses = {f: 0 for f in active}
    boards, seen = [], set()
    turn = 0
    while len(boards) < n and active:
        family = active[turn % len(active)]
        turn += 1
        desc, code = FAMILIES[family](rng)
        if desc in seen or normalize(desc) in held_out:
            misses[family] += 1
            if misses[family] >= max_misses:
                active.remove(family)
            continue
        misses[family] = 0
        seen.add(desc)
        boards.append({"id": f"{family}_{len(boards):04d}", "family": family,
                       "description": desc, "code": code})
    return boards


# ----------------------------------------------------------------------------
# Verification and export
# ----------------------------------------------------------------------------

def verify(board, workdir):
    """Run the board's code through ERC, structural checks and its requirements."""
    from .checks import check_circuit
    from .runner import run_circuit_code
    run = run_circuit_code(board["code"], workdir)
    if not run.ok:
        return False, run.summary(), None
    report = check_circuit(run.circuit)
    feedback = run.summary() + "\n" + report.summary()
    passed = report.ok and not run.erc_errors
    return passed, ("PASSED. " if passed else "FAILED. ") + feedback, run.circuit


_VALUE = re.compile(r'\b([RC])\("([^"]+)"(?=[,)])')
_FAILED_REQ = re.compile(r"requirement not met: (.+?): FAIL measured (\S+) .*?required (\S+)")


def plant_bug(board, workdir, rng):
    """A copy of the board with one R or C value off by a factor of ten that fails a requirement.

    Returns {"code", "feedback", "diagnosis"} with the real check_circuit
    output for the broken code, or None if no single change breaks a requirement.
    """
    from .spice import parse_value
    code = board["code"]
    matches = list(_VALUE.finditer(code))
    rng.shuffle(matches)
    for m in matches:
        kind, good = m.group(1), m.group(2)
        for factor in rng.sample([10, 0.1], 2):
            bad = fmt(parse_value(good) * factor)
            broken = code[:m.start()] + f'{kind}("{bad}"' + code[m.end():]
            passed, feedback, _ = verify({**board, "code": broken}, workdir)
            failure = _FAILED_REQ.search(feedback)
            if passed or not failure:
                continue
            part = "resistor" if kind == "R" else "capacitor"
            name, measured, required = failure.groups()
            diagnosis = (f'The requirement "{name}" fails: measured {measured}, required '
                         f'{required}. The {bad} {part} is a factor of ten off; from the design '
                         f'calculation it should be {good}. Fixing it and checking again.')
            return {"code": broken, "feedback": feedback, "diagnosis": diagnosis}
    return None


def _parts_used(circuit):
    seen, out = set(), []
    for p in circuit["parts"]:
        lib, name = p["symbol"].split(":", 1)
        if (lib, name) not in seen:
            seen.add((lib, name))
            out.append((lib, name, p["footprint"]))
    return out


def tool_session(board, circuit, feedback, repair=None):
    """A tool-using design session for this board, built from real tool calls.

    Mirrors what a good designer does: look up each part, read its pins, then
    check the circuit and summarise. Tool results are produced by the same
    functions the live tools call. With `repair` (from plant_bug), the first
    check fails on a planted bug and the next turn diagnoses it and resubmits
    the fixed code in the same message.
    """
    from .generate import TOOLS, Toolbox, _system_prompt
    from pathlib import Path
    import json as _json
    import tempfile

    tools = Toolbox(Path(tempfile.mkdtemp()))
    messages = [{"role": "system", "content": _system_prompt()},
                {"role": "user", "content": f"Design this board:\n\n{board['description']}"}]
    n = 0

    def call(name, args, text="", result=None):
        nonlocal n
        n += 1
        cid = f"call_{n}"
        messages.append({"role": "assistant", "content": text, "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": _json.dumps(args)}}]})
        if result is not None:
            content = result
        elif name == "check_circuit":
            content = feedback
        else:
            content, is_error = tools.execute(name, args)
            if is_error:
                content = f"ERROR: {content}"
        messages.append({"role": "tool", "tool_call_id": cid, "content": content})

    for lib, name, _ in _parts_used(circuit):
        if lib == "Device" and name in ("R", "C"):
            continue
        call("search_symbols", {"query": name})
        call("get_symbol", {"lib": lib, "name": name})
    if repair:
        call("check_circuit", {"code": repair["code"]}, result=repair["feedback"])
        call("check_circuit", {"code": board["code"]}, text=repair["diagnosis"])
    else:
        call("check_circuit", {"code": board["code"]})
    summary = board["code"].split('"""')[1].strip() if board["code"].startswith('"""') else ""
    messages.append({"role": "assistant",
                     "content": f"The design passes check_circuit, including its requirements "
                                f"across component tolerances.\n\n{summary}"})
    hf_tools = [{"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in TOOLS]
    return {"messages": messages, "tools": hf_tools}


def _verify_one(args):
    board, workdir, repair_fraction, seed = args
    passed, feedback, circuit = verify(board, workdir)
    repair = None
    rng = random.Random(f"{seed}-{board['id']}")
    if passed and rng.random() < repair_fraction:
        repair = plant_bug(board, workdir + "_bug", rng)
    return board, passed, feedback, circuit, repair


def build_dataset(n, out_dir, seed=0, jobs=4, repair_fraction=0.0, families=None):
    """Generate n boards, verify them in parallel, export the ones that pass. Returns stats.

    A `repair_fraction` of the agentic sessions start from a planted bug
    (plant_bug) and show the diagnosis and fix.
    """
    import json as _json
    import multiprocessing
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    from .datagen import DIRECT_SYSTEM

    out_dir = Path(out_dir)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    boards = generate(n, seed=seed, families=families)
    kept, failed = [], Counter()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        for i, (board, passed, feedback, circuit, repair) in enumerate(pool.map(
                _verify_one, [(b, str(work / b["id"]), repair_fraction, seed) for b in boards]), 1):
            if passed:
                kept.append((board, feedback, circuit, repair))
            else:
                failed[board["family"]] += 1
                (work / board["id"] / "FAILED.txt").write_text(feedback)
            if i % 50 == 0:
                print(f"verified {i}/{len(boards)}: kept {len(kept)}", flush=True)

    with open(out_dir / "boards.jsonl", "w") as f_boards, \
            open(out_dir / "sft_direct.jsonl", "w") as f_direct, \
            open(out_dir / "sft_agentic.jsonl", "w") as f_agentic:
        for board, feedback, circuit, repair in kept:
            meta = {"board": board["id"], "family": board["family"], "source": "procgen",
                    "repair": repair is not None}
            f_boards.write(_json.dumps(board) + "\n")
            f_direct.write(_json.dumps({"messages": [
                {"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": board["description"]},
                {"role": "assistant", "content": f"```python\n{board['code'].strip()}\n```"},
            ], **meta}) + "\n")
            f_agentic.write(_json.dumps({**tool_session(board, circuit, feedback, repair),
                                         **meta}) + "\n")
    stats = {"generated": len(boards), "kept": len(kept),
             "repair_sessions": sum(r is not None for *_, r in kept),
             "kept_by_family": dict(Counter(b["family"] for b, *_ in kept)),
             "failed_by_family": dict(failed)}
    (out_dir / "stats.json").write_text(_json.dumps(stats, indent=2))
    return stats
