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

    @property
    def ok(self):
        return not self.errors

    def summary(self):
        if not self.errors and not self.warnings:
            return "Design checks passed."
        lines = [f"DESIGN ERROR: {e}" for e in self.errors]
        lines += [f"DESIGN WARNING: {w}" for w in self.warnings]
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


def check_circuit(circuit):
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
    return report
