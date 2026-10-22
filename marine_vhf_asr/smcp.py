"""
Synthetic marine VHF dialogue generator.

Generates multi-turn radio exchanges in the style of the IMO Standard Marine
Communication Phrases (SMCP) and common VTS / port practice: entry reports,
pilot requests, berthing instructions, MAYDAY / PAN PAN / SECURITE calls,
ship-to-ship passing arrangements and radio checks.

Each turn is one push-to-talk transmission. Turn text is written the way it
is spoken (numbers as words) so it can be fed straight to a TTS engine, and
the training target is simply normalize_transcript(text).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .gazetteer import (
    DISTRESS_NATURE, PORTS, SECURITE_NATURE, URGENCY_NATURE,
    VESSEL_PREFIXES, VESSEL_TYPES, VESSEL_WORDS, Port,
)
from .normalize import DIGITS, PHONETIC, normalize_transcript

_SMALL = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


@dataclass
class Turn:
    """One radio transmission."""
    speaker: str                 # "vessel", "shore" or "vessel2"
    text: str                    # spoken form, for TTS
    slots: Dict[str, str] = field(default_factory=dict)  # critical values, spoken form

    @property
    def target(self) -> str:
        return normalize_transcript(self.text)


@dataclass
class Dialogue:
    scenario: str
    port: str
    turns: List[Turn]


def number_words(n: int) -> str:
    """Natural English number for 0-99: 42 -> 'forty two'."""
    if n < 20:
        return _SMALL[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f" {_SMALL[ones]}" if ones else "")


def digits(s: str) -> str:
    """Digit-by-digit spoken form: '074' -> 'zero seven four'."""
    return " ".join(DIGITS[d].lower() for d in s)


def phonetic(s: str) -> str:
    """Phonetic spelling: 'PD2' -> 'Papa Delta two'."""
    return " ".join(PHONETIC[c].capitalize() if c in PHONETIC else DIGITS[c].lower() for c in s)


class DialogueGenerator:
    """
    Random SMCP-style dialogue generator.

    Args:
        seed: RNG seed
        ports: Ports to draw from (defaults to gazetteer.PORTS)
    """

    def __init__(self, seed: Optional[int] = None, ports: Optional[List[Port]] = None):
        self.rng = np.random.default_rng(seed)
        self.ports = ports or PORTS
        self.scenarios: Dict[str, Callable[[Port], List[Turn]]] = {
            "vts_entry_report": self._vts_entry_report,
            "pilot_request": self._pilot_request,
            "berthing": self._berthing,
            "mayday": self._mayday,
            "pan_pan": self._pan_pan,
            "securite": self._securite,
            "ship_to_ship": self._ship_to_ship,
            "radio_check": self._radio_check,
        }

    # ------------------------------------------------------------------ helpers

    def _choice(self, items):
        return items[self.rng.integers(len(items))]

    def _p(self, p: float) -> bool:
        return self.rng.random() < p

    def _vessel(self) -> str:
        return (self._choice(VESSEL_PREFIXES) + self._choice(VESSEL_WORDS)).strip()

    def _callsign(self) -> str:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        cs = "".join(self._choice(letters) for _ in range(2))
        cs += "".join(self._choice(letters + "0123456789") for _ in range(self.rng.integers(2, 4)))
        return phonetic(cs)

    def _mmsi(self, mid: str) -> str:
        return digits(mid + "".join(str(self.rng.integers(10)) for _ in range(6)))

    def _channel(self, ch: str) -> str:
        if self._p(0.25) and len(ch) == 2 and int(ch) < 20:
            return f"channel {number_words(int(ch))}"
        return f"channel {digits(ch.zfill(2))}"

    def _decimal(self) -> str:
        return "decimal" if self._p(0.5) else "point"

    def _time(self) -> str:
        hh, mm = int(self.rng.integers(24)), int(self.rng.integers(12)) * 5
        return f"{digits(f'{hh:02d}{mm:02d}')} UTC"

    def _position(self, port: Port) -> str:
        """Lat/lon a few miles off the port, SMCP style."""
        def coord(deg, minutes, hemi, width, offset):
            total = deg * 60 + minutes + offset
            d, m = divmod(abs(total), 60)
            tenths = self.rng.integers(10)
            hemi_word = {"N": "north", "S": "south", "E": "east", "W": "west"}[hemi]
            return (f"{digits(f'{int(d):0{width}d}')} degrees "
                    f"{digits(f'{int(m):02d}')} {self._decimal()} {digits(str(tenths))} minutes "
                    f"{hemi_word}")

        lat = coord(*port.lat, 2, int(self.rng.integers(-20, 21)))
        lon = coord(*port.lon, 3, int(self.rng.integers(-20, 21)))
        return f"{lat}, {lon}"

    def _call(self, to: str, frm: str) -> str:
        """Initial call: 'X, X, this is Y, Y'."""
        to_part = f"{to}, {to}" if self._p(0.6) else to
        frm_part = f"{frm}, {frm}" if self._p(0.5) else frm
        return f"{to_part}, this is {frm_part}"

    def _draught(self) -> str:
        whole = int(self.rng.integers(3, 16))
        return f"{number_words(whole)} {self._decimal()} {_SMALL[self.rng.integers(10)]} metres"

    # ---------------------------------------------------------------- scenarios

    def _vts_entry_report(self, port: Port) -> List[Turn]:
        vessel = self._vessel()
        cs = self._callsign()
        ch = self._channel(self._choice(port.vts_channels))
        eta = self._time()
        draught = self._draught()
        persons = number_words(int(self.rng.integers(8, 30)))
        return [
            Turn("vessel", f"{self._call(port.vts, vessel)}, {ch}, over.",
                 {"station": port.vts, "vessel": vessel}),
            Turn("shore", f"{vessel}, {port.vts}, go ahead, over.", {"vessel": vessel}),
            Turn("vessel",
                 f"{port.vts}, {vessel}, call sign {cs}. Inbound {port.name}, "
                 f"ETA pilot station {eta}. Maximum draught {draught}. "
                 f"{persons.capitalize()} persons on board, no defects, over.",
                 {"callsign": cs, "eta": eta}),
            Turn("shore",
                 f"{vessel}, {port.vts}. Information received. "
                 f"Report again at the fairway buoy on {ch}, over.",
                 {"channel": ch.split(" ", 1)[1]}),
            Turn("vessel", f"Report at the fairway buoy, {vessel}, out.", {"vessel": vessel}),
        ]

    def _pilot_request(self, port: Port) -> List[Turn]:
        vessel = self._vessel()
        eta = self._time()
        side = self._choice(["port", "starboard"])
        ladder = self._choice(["pilot ladder", "combination ladder"])
        height = number_words(int(self.rng.integers(1, 10)))
        return [
            Turn("vessel", f"{self._call(port.pilots, vessel)}. REQUEST: pilot. "
                           f"My ETA at pilot boarding position {eta}, over.",
                 {"station": port.pilots, "eta": eta}),
            Turn("shore", f"{vessel}, {port.pilots}. Pilot will board at {eta}. "
                          f"Rig {ladder} on {side} side, {height} metres above water, over.",
                 {"side": side}),
            Turn("vessel", f"{ladder.capitalize()} on {side} side, {height} metres above water. "
                           f"{vessel}, out.", {"side": side}),
        ]

    def _berthing(self, port: Port) -> List[Turn]:
        vessel = self._vessel()
        berth = self._choice(port.berths) if port.berths else "the inner harbour"
        ch = self._channel(self._choice(port.vts_channels))
        tugs = number_words(int(self.rng.integers(1, 4)))
        side = self._choice(["port", "starboard"])
        return [
            Turn("shore", f"{vessel}, {port.vts}. INSTRUCTION: proceed to {berth}, "
                          f"{side} side alongside. {tugs.capitalize()} tugs will meet you "
                          f"at the breakwater, over.",
                 {"berth": berth, "side": side}),
            Turn("vessel", f"{port.vts}, {vessel}. Proceed to {berth}, {side} side alongside, "
                           f"{tugs} tugs at the breakwater, over.", {"berth": berth}),
            Turn("shore", f"{vessel}, correct. Switch to {ch} for berthing, over.",
                 {"channel": ch.split(" ", 1)[1]}),
            Turn("vessel", f"Switching to {ch}, {vessel}.", {"vessel": vessel}),
        ]

    def _mayday(self, port: Port) -> List[Turn]:
        vessel = self._vessel()
        cs = self._callsign()
        mmsi = self._mmsi(port.mid)
        pos = self._position(port)
        nature = self._choice(DISTRESS_NATURE)
        persons = number_words(int(self.rng.integers(1, 40)))
        vtype = self._choice(VESSEL_TYPES)
        return [
            Turn("vessel",
                 f"MAYDAY, MAYDAY, MAYDAY. This is {vessel}, {vessel}, {vessel}, "
                 f"call sign {cs}, MMSI {mmsi}. MAYDAY {vessel}. Position {pos}. "
                 f"{nature}. I require immediate assistance. "
                 f"{persons.capitalize()} persons on board. {vtype.capitalize()}, over.",
                 {"vessel": vessel, "mmsi": mmsi, "position": pos}),
            Turn("shore",
                 f"MAYDAY, {vessel}, {vessel}, {vessel}, this is {port.coastguard}, "
                 f"{port.coastguard}, {port.coastguard}. Received MAYDAY. "
                 f"Lifeboat is launching, ETA {number_words(int(self.rng.integers(10, 60)))} minutes, over.",
                 {"vessel": vessel, "station": port.coastguard}),
        ]

    def _pan_pan(self, port: Port) -> List[Turn]:
        vessel = self._vessel()
        pos = self._position(port)
        nature = self._choice(URGENCY_NATURE)
        return [
            Turn("vessel",
                 f"PAN PAN, PAN PAN, PAN PAN. All stations, all stations, all stations. "
                 f"This is {vessel}, {vessel}, {vessel}. Position {pos}. "
                 f"{nature}. I require assistance, over.",
                 {"vessel": vessel, "position": pos}),
            Turn("shore", f"{vessel}, this is {port.coastguard}. Received PAN PAN. "
                          f"Stand by on {self._channel('16')}, over.",
                 {"station": port.coastguard}),
        ]

    def _securite(self, port: Port) -> List[Turn]:
        nature = self._choice(SECURITE_NATURE)
        pos = self._position(port)
        ch = self._channel(self._choice(["62", "67", "10", "23", "86"]))
        return [
            Turn("shore",
                 f"SECURITE, SECURITE, SECURITE. All stations, all stations, all stations. "
                 f"This is {port.coastguard}. For navigational warning listen on "
                 f"{ch}, out.",
                 {"station": port.coastguard, "channel": ch.split(" ", 1)[1]}),
            Turn("shore",
                 f"SECURITE, this is {port.coastguard}. Navigational warning. "
                 f"{nature.capitalize()} in position {pos}. Vessels are advised to keep "
                 f"well clear. {port.coastguard}, out.",
                 {"position": pos}),
        ]

    def _ship_to_ship(self, port: Port) -> List[Turn]:
        v1, v2 = self._vessel(), self._vessel()
        side = self._choice(["port to port", "starboard to starboard"])
        overtake = self._choice(["port", "starboard"])
        if self._p(0.5):
            ask = f"QUESTION: do you agree to pass {side}, over."
            reply = f"{v1}, {v2}. ANSWER: yes, I agree to pass {side}, over."
            slots = {"manoeuvre": side}
        else:
            ask = f"INTENTION: I intend to overtake you on your {overtake} side, over."
            reply = f"{v1}, {v2}. Agree, overtake on my {overtake} side, over."
            slots = {"manoeuvre": overtake}
        return [
            Turn("vessel", f"{self._call(v2, v1)}, {self._channel('13')}, over.",
                 {"vessel": v2}),
            Turn("vessel2", f"{v1}, this is {v2}, go ahead, over.", {"vessel": v1}),
            Turn("vessel", f"{v2}, {v1}. {ask}", slots),
            Turn("vessel2", reply, slots),
            Turn("vessel", f"{v1}, out.", {"vessel": v1}),
        ]

    def _radio_check(self, port: Port) -> List[Turn]:
        vessel = self._vessel()
        ch = self._choice(["16", "67", "12"])
        quality = self._choice(["loud and clear", "readability five", "readability three, broken"])
        return [
            Turn("vessel", f"{self._call(port.coastguard, vessel)}, radio check "
                           f"{self._channel(ch)}, over.",
                 {"station": port.coastguard, "vessel": vessel}),
            Turn("shore", f"{vessel}, {port.coastguard}. I read you {quality}, over.",
                 {"vessel": vessel}),
            Turn("vessel", f"Thank you, {vessel}, out.", {"vessel": vessel}),
        ]

    # --------------------------------------------------------------------- API

    def generate(self, scenario: Optional[str] = None, port: Optional[Port] = None) -> Dialogue:
        """Generate one dialogue, choosing scenario and port at random if not given."""
        scenario = scenario or self._choice(list(self.scenarios))
        port = port or self._choice(self.ports)
        return Dialogue(scenario, port.name, self.scenarios[scenario](port))
