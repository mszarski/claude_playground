"""
Design checks on a circuit dumped by the runner.

These cover the mistakes that SKiDL's ERC does not catch but that stop a
board from being built or from working: missing or wrong footprints,
dangling nets, and ICs without decoupling capacitors.
"""

import re
from dataclasses import dataclass, field

from . import library

GROUND_NET = re.compile(r"^(A|D|P)?(GND|VSS|0V)\w*$", re.IGNORECASE)
NO_CONNECT_TYPES = {"NOCONNECT", "no_connect"}
POWER_IN_TYPES = {"PWRIN", "power_in"}


@dataclass
class CheckReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)      # e.g. requirements that passed

    @property
    def ok(self):
        return not self.errors

    def summary(self):
        lines = [f"DESIGN ERROR: {e}" for e in self.errors]
        lines += [f"DESIGN WARNING: {w}" for w in self.warnings]
        lines += [f"REQUIREMENT OK: {n}" for n in self.notes]
        if not self.errors and not self.warnings:
            lines.insert(0, "Design checks passed.")
        return "\n".join(lines)


def is_ground(net):
    return bool(net) and bool(GROUND_NET.match(net))


def _net_members(circuit):
    members = {}
    for part in circuit["parts"]:
        for pin in part["pins"]:
            if pin["net"]:
                members.setdefault(pin["net"], []).append((part["ref"], pin["num"]))
    return members


def _is_capacitor(part):
    return part["ref"].startswith("C") and len(part["pins"]) == 2


def _is_ic(part):
    return part["ref"].startswith("U")


# Pin types that set a net's DC level, and the ones that must not be left floating.
DRIVING_TYPES = {"OUTPUT", "PWROUT", "BIDIR", "TRISTATE", "output", "power_out", "bidirectional",
                 "tri_state"}
INPUT_TYPES = {"INPUT", "input"}
PUSH_PULL_OUTPUT_TYPES = {"OUTPUT", "PWROUT", "output", "power_out"}
# Two-terminal parts that conduct at DC: resistors, inductors, ferrites, fuses, diodes,
# jumpers. Crystals are included because oscillator pins are biased inside the IC.
DC_CONDUCTING_PREFIXES = ("R", "L", "FB", "F", "D", "JP", "Y")
# Input pins that are biased inside the part, so a capacitor alone is fine.
INTERNALLY_BIASED = [(re.compile(r"(^|[^0-9])55[56]"), {"CV"})]


def _internally_biased(part, pin):
    name = part["symbol"].split(":", 1)[-1]
    return any(pat.search(name) and pin["name"] in pins for pat, pins in INTERNALLY_BIASED)


def _ref_prefix(ref):
    return re.match(r"[A-Za-z]+", ref).group(0) if re.match(r"[A-Za-z]+", ref) else ""


def _check_dc_paths(parts, members, report):
    """Every net with an input pin needs a DC path to something that sets its voltage."""
    defined = {n for n in members if is_ground(n)}
    for part in parts:
        for pin in part["pins"]:
            if not pin["net"]:
                continue
            # Connectors bring signals and supplies from outside the board.
            if part["ref"].startswith("J") or pin["type"] in DRIVING_TYPES:
                defined.add(pin["net"])

    edges = {}
    for part in parts:
        prefix = _ref_prefix(part["ref"])
        nets = [p["net"] for p in part["pins"] if p["net"]]
        if prefix in DC_CONDUCTING_PREFIXES and len(part["pins"]) == 2 and len(nets) == 2:
            a, b = nets
            edges.setdefault(a, set()).add(b)
            edges.setdefault(b, set()).add(a)

    frontier = list(defined)
    while frontier:
        net = frontier.pop()
        for other in edges.get(net, ()):
            if other not in defined:
                defined.add(other)
                frontier.append(other)

    for net, pins in members.items():
        if net in defined:
            continue
        inputs = []
        for ref, num in pins:
            part = next(p for p in parts if p["ref"] == ref)
            pin = _pin(parts, ref, num)
            if pin["type"] in INPUT_TYPES and not _internally_biased(part, pin):
                inputs.append(f"{ref} pin {num}")
        if inputs:
            report.errors.append(
                f"net '{net}' has no DC path to a supply, ground, output or connector, so "
                f"the input(s) on it float ({', '.join(inputs)}). Add a bias or pull resistor "
                f"(capacitors do not set a DC level).")


def _check_outputs_on_rails(parts, members, report):
    """A push-pull output wired straight to ground or to a supply rail fights that rail."""
    rails = set()
    for net, pins in members.items():
        if is_ground(net):
            rails.add(net)
        elif any(_pin(parts, r, n)["type"] in {"PWROUT", "power_out"} for r, n in pins):
            rails.add(net)
    for part in parts:
        for pin in part["pins"]:
            if pin["type"] in {"OUTPUT", "output"} and pin["net"] in rails:
                report.errors.append(
                    f"{part['ref']} pin {pin['num']} ({pin['name']}) is an output tied directly "
                    f"to rail '{pin['net']}'. Outputs must not drive a supply or ground; for an "
                    f"unused op-amp section, connect it as a follower (output to - input) with "
                    f"+ at a defined level.")


def _pin(parts, ref, num):
    part = next(p for p in parts if p["ref"] == ref)
    return next(p for p in part["pins"] if p["num"] == num)


def _check_requirements(circuit, report):
    """Simulate each require(...) at nominal values and across component tolerances."""
    from .spice import requirement_tests, simulate
    tests = requirement_tests(circuit)
    if not tests:
        report.warnings.append(
            "no require(...) declarations: state the behaviour the description asks for "
            "(currents, voltages, gains) so it is checked in simulation")
        return
    sim = simulate(circuit, tests, corners=True)
    for result in sim.results:
        if result.passed:
            report.notes.append(result.describe())
        else:
            report.errors.append(f"requirement not met: {result.describe()}")
    if sim.unmodeled and not sim.ok:
        report.warnings.append("parts without a simulation model (left out): "
                               + ", ".join(sim.unmodeled))


def check_circuit(circuit, requirements=True):
    """Structural checks, plus the circuit's own requirements in simulation."""
    report = CheckReport()
    parts = circuit["parts"]
    if not parts:
        report.errors.append("circuit has no parts")
        return report

    for part in parts:
        ref, fp = part["ref"], part["footprint"]
        if not fp:
            report.errors.append(f"{ref} ({part['symbol']}) has no footprint")
        elif not library.footprint_exists(fp):
            report.errors.append(f"{ref}: footprint '{fp}' is not in the KiCad library")
        else:
            pads = library.footprint_pads(fp)
            missing = sorted({p["num"] for p in part["pins"]} - pads)
            if missing:
                report.errors.append(
                    f"{ref}: footprint '{fp}' has no pad(s) {', '.join(missing)} "
                    f"for symbol pins; pick a footprint that matches the symbol's pin numbers")

        for pin in part["pins"]:
            if pin["net"] is None and pin["type"] not in NO_CONNECT_TYPES:
                report.warnings.append(f"{ref} pin {pin['num']} ({pin['name']}) is unconnected")

    members = _net_members(circuit)
    for net, pins in members.items():
        if len(pins) == 1:
            ref, num = pins[0]
            report.errors.append(f"net '{net}' only connects to {ref} pin {num}")

    if not any(is_ground(n) for n in members):
        report.warnings.append("no ground net found (expected a net named GND)")

    _check_dc_paths(parts, members, report)
    _check_outputs_on_rails(parts, members, report)

    # Every IC power-input net (other than ground) needs a capacitor to ground.
    decoupled = set()
    for part in parts:
        if _is_capacitor(part):
            nets = [p["net"] for p in part["pins"]]
            if any(is_ground(n) for n in nets):
                decoupled.update(n for n in nets if n and not is_ground(n))
    for part in parts:
        if not _is_ic(part):
            continue
        for pin in part["pins"]:
            net = pin["net"]
            if pin["type"] in POWER_IN_TYPES and net and not is_ground(net) and net not in decoupled:
                report.errors.append(
                    f"{part['ref']} power pin {pin['num']} ({pin['name']}) on net '{net}' "
                    f"has no decoupling capacitor to ground")
    if requirements and not report.errors:
        _check_requirements(circuit, report)
    return report
