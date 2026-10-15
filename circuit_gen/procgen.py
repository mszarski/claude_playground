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
}


def generate(n, seed=0, families=None, max_misses=50):
    """n boards, round-robin over families. Returns dicts with id, family, description, code.

    A family that keeps producing descriptions already seen is retired, so a
    small parameter space cannot stall the run.
    """
    rng = random.Random(seed)
    active = list(families or FAMILIES)
    misses = {f: 0 for f in active}
    boards, seen = [], set()
    turn = 0
    while len(boards) < n and active:
        family = active[turn % len(active)]
        turn += 1
        desc, code = FAMILIES[family](rng)
        if desc in seen:
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


def _parts_used(circuit):
    seen, out = set(), []
    for p in circuit["parts"]:
        lib, name = p["symbol"].split(":", 1)
        if (lib, name) not in seen:
            seen.add((lib, name))
            out.append((lib, name, p["footprint"]))
    return out


def tool_session(board, circuit, feedback):
    """A tool-using design session for this board, built from real tool calls.

    Mirrors what a good designer does: look up each part, read its pins, then
    check the circuit once and summarise. Tool results are produced by the
    same functions the live tools call.
    """
    from .generate import TOOLS, Toolbox, _system_prompt
    from pathlib import Path
    import json as _json
    import tempfile

    tools = Toolbox(Path(tempfile.mkdtemp()))
    messages = [{"role": "system", "content": _system_prompt()},
                {"role": "user", "content": f"Design this board:\n\n{board['description']}"}]
    n = 0

    def call(name, args):
        nonlocal n
        n += 1
        cid = f"call_{n}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": _json.dumps(args)}}]})
        if name == "check_circuit":
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
    board, workdir = args
    passed, feedback, circuit = verify(board, workdir)
    return board, passed, feedback, circuit


def build_dataset(n, out_dir, seed=0, jobs=4):
    """Generate n boards, verify them in parallel, export the ones that pass. Returns stats."""
    import json as _json
    import multiprocessing
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    from .datagen import DIRECT_SYSTEM

    out_dir = Path(out_dir)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    boards = generate(n, seed=seed)
    kept, failed = [], Counter()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        for i, (board, passed, feedback, circuit) in enumerate(pool.map(
                _verify_one, [(b, str(work / b["id"])) for b in boards]), 1):
            if passed:
                kept.append((board, feedback, circuit))
            else:
                failed[board["family"]] += 1
                (work / board["id"] / "FAILED.txt").write_text(feedback)
            if i % 50 == 0:
                print(f"verified {i}/{len(boards)}: kept {len(kept)}", flush=True)

    with open(out_dir / "boards.jsonl", "w") as f_boards, \
            open(out_dir / "sft_direct.jsonl", "w") as f_direct, \
            open(out_dir / "sft_agentic.jsonl", "w") as f_agentic:
        for board, feedback, circuit in kept:
            meta = {"board": board["id"], "family": board["family"], "source": "procgen"}
            f_boards.write(_json.dumps(board) + "\n")
            f_direct.write(_json.dumps({"messages": [
                {"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": board["description"]},
                {"role": "assistant", "content": f"```python\n{board['code'].strip()}\n```"},
            ], **meta}) + "\n")
            f_agentic.write(_json.dumps({**tool_session(board, circuit, feedback), **meta}) + "\n")
    stats = {"generated": len(boards), "kept": len(kept),
             "kept_by_family": dict(Counter(b["family"] for b, _, _ in kept)),
             "failed_by_family": dict(failed)}
    (out_dir / "stats.json").write_text(_json.dumps(stats, indent=2))
    return stats
