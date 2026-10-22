"""
Named entities used to fill the SMCP dialogue templates.

Replace or extend these lists with the ports, stations and traffic you care
about: the transcription model can only learn names it has heard.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Port:
    name: str
    vts: str                       # VTS / harbour control station name as spoken
    pilots: str                    # pilot station name as spoken
    coastguard: str                # nearest MRCC / coastguard radio
    mid: str                       # flag MID most common for local traffic
    lat: Tuple[int, int, str]      # approx. (degrees, minutes, hemisphere)
    lon: Tuple[int, int, str]
    vts_channels: Tuple[str, ...]  # working channels, as digit strings
    berths: Tuple[str, ...] = ()


PORTS = [
    Port("Rotterdam", "Maas Entrance", "Rotterdam Pilots", "Den Helder Coastguard", "244",
         (51, 59, "N"), (4, 3, "E"), ("3", "19", "1"), ("Maasvlakte", "Europoort", "Waalhaven")),
    Port("Antwerp", "Zeebrugge Traffic Centre", "Wandelaar Pilots", "MRCC Ostend", "205",
         (51, 21, "N"), (3, 11, "E"), ("69", "12"), ("Deurganck Dock", "Delwaide Dock")),
    Port("Hamburg", "Hamburg Port Traffic", "Elbe Pilot", "Bremen Rescue", "211",
         (53, 32, "N"), (9, 58, "E"), ("74", "14", "19"), ("Burchardkai", "Eurogate", "Tollerort")),
    Port("Felixstowe", "Harwich VTS", "Sunk Pilots", "Humber Coastguard", "235",
         (51, 57, "N"), (1, 19, "E"), ("71", "14"), ("Trinity Terminal", "Berth Eight")),
    Port("Southampton", "Southampton VTS", "Southampton Pilots", "Solent Coastguard", "235",
         (50, 53, "N"), (1, 24, "W"), ("12", "14"), ("Ocean Terminal", "Mayflower Terminal")),
    Port("Dover", "Dover Port Control", "Dover Pilots", "Dover Coastguard", "235",
         (51, 7, "N"), (1, 20, "E"), ("74", "12"), ("Eastern Docks", "Cruise Terminal One")),
    Port("Le Havre", "Le Havre Port Control", "Le Havre Pilots", "CROSS Gris-Nez", "227",
         (49, 29, "N"), (0, 6, "E"), ("67", "12"), ("Port Two Thousand",)),
    Port("Dublin", "Dublin VTS", "Dublin Pilots", "Dublin Coast Guard", "250",
         (53, 21, "N"), (6, 13, "W"), ("12",), ("Ocean Pier", "Alexandra Basin")),
    Port("Cork", "Cork Harbour Radio", "Cork Pilots", "Valentia Coast Guard", "250",
         (51, 50, "N"), (8, 17, "W"), ("12", "14"), ("Ringaskiddy",)),
    Port("Gothenburg", "Gothenburg VTS", "Gothenburg Pilots", "Sweden Rescue", "265",
         (57, 41, "N"), (11, 51, "E"), ("13", "16"), ("Skandia Harbour",)),
    Port("Gdansk", "Gdansk VTS", "Gdansk Pilots", "Gdynia Rescue", "261",
         (54, 24, "N"), (18, 40, "E"), ("71", "14"), ("Deepwater Container Terminal",)),
    Port("Valencia", "Valencia Traffic", "Valencia Pilots", "Valencia Rescue", "224",
         (39, 26, "N"), (0, 19, "W"), ("14", "11"), ("Principe Felipe Terminal",)),
    Port("Piraeus", "Piraeus Traffic", "Piraeus Pilots", "Piraeus JRCC", "237",
         (37, 56, "N"), (23, 37, "E"), ("12", "14"), ("Pier Two", "Pier Three")),
    Port("Singapore", "Singapore VTIS Central", "Singapore Pilots", "MRCC Singapore", "563",
         (1, 16, "N"), (103, 50, "E"), ("14", "12", "73"), ("Pasir Panjang", "Tanjong Pagar")),
    Port("Los Angeles", "LA Long Beach Vessel Traffic", "Los Angeles Pilots", "Coast Guard Sector Los Angeles", "366",
         (33, 43, "N"), (118, 16, "W"), ("14", "73"), ("Pier Four Hundred", "Terminal Island")),
    Port("New York", "New York Traffic", "Sandy Hook Pilots", "Coast Guard Sector New York", "366",
         (40, 35, "N"), (74, 2, "W"), ("14", "12"), ("Port Newark", "Red Hook")),
    Port("Halifax", "Halifax Traffic", "Halifax Pilots", "Halifax Coast Guard Radio", "316",
         (44, 38, "N"), (63, 34, "W"), ("14", "12"), ("Fairview Cove", "South End Terminal")),
    Port("Vancouver", "Victoria Traffic", "Pacific Pilots", "Prince Rupert Coast Guard Radio", "316",
         (49, 17, "N"), (123, 6, "W"), ("12", "11"), ("Centerm", "Deltaport")),
    Port("Sydney", "Sydney VTS", "Sydney Pilots", "JRCC Australia", "503",
         (33, 50, "S"), (151, 16, "E"), ("13", "12"), ("Port Botany", "White Bay")),
]

VESSEL_PREFIXES = ["", "", "", "MV ", "MT ", "MSC ", "Maersk ", "CMA CGM ", "Ever ", "Stena ", "Nordic ", "Atlantic "]

VESSEL_WORDS = [
    "Aurora", "Horizon", "Pioneer", "Northern Star", "Sea Breeze", "Endeavour", "Valiant",
    "Orion", "Neptune", "Polaris", "Liberty", "Spirit", "Harmony", "Explorer", "Mariner",
    "Seahawk", "Osprey", "Kittiwake", "Cormorant", "Albatross", "Fulmar", "Gannet",
    "Galway Bay", "Baltic Trader", "Ocean Glory", "Silver Wave", "Blue Marlin", "Red Kite",
    "Atlas", "Hercules", "Zeus", "Athena", "Elena", "Sofia", "Isabella", "Anna Maria",
    "Golden Dawn", "Morning Tide", "Westward", "Southern Cross", "Celtic Explorer",
]

VESSEL_TYPES = [
    "container vessel", "bulk carrier", "tanker", "general cargo vessel", "ro-ro ferry",
    "passenger vessel", "fishing vessel", "sailing yacht", "motor yacht", "tug", "dredger",
]

DISTRESS_NATURE = [
    "I am on fire", "I am taking on water", "I am aground", "I am sinking",
    "I am listing heavily", "I have had a collision", "I have flooding in the engine room",
    "I have capsized",
]

URGENCY_NATURE = [
    "I have engine failure and am drifting", "I have a man overboard", "I have steering failure",
    "I have a medical emergency on board", "I have a fouled propeller", "I have lost power",
]

SECURITE_NATURE = [
    "unlit buoy", "buoy off station", "large floating debris", "drifting container",
    "dangerous wreck", "dredging operations in progress",
    "vessel restricted in her ability to manoeuvre",
    "submarine cable laying operations", "diving operations in progress",
]
