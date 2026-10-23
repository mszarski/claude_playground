"""
Named entities used to fill the SMCP dialogue templates.

Replace or extend these lists with the ports, stations and traffic you care
about: the transcription model can only learn names it has heard.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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
    receiver: Optional[Tuple[float, float]] = None  # (lat, lon) of the listening station, for AIS snapshots


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
         (33, 43, "N"), (118, 16, "W"), ("14", "73"), ("Pier Four Hundred", "Terminal Island"), (33.72, -118.27)),
    Port("New York", "New York Traffic", "Sandy Hook Pilots", "Coast Guard Sector New York", "366",
         (40, 35, "N"), (74, 2, "W"), ("14", "12", "11"),
         ("Port Newark", "Port Elizabeth", "Red Hook", "Howland Hook", "GCT Bayonne",
          "Global Terminal", "Stapleton Anchorage", "Gravesend Anchorage"), (40.61, -74.06)),
    Port("Halifax", "Halifax Traffic", "Halifax Pilots", "Halifax Coast Guard Radio", "316",
         (44, 38, "N"), (63, 34, "W"), ("14", "12"), ("Fairview Cove", "South End Terminal")),
    Port("Vancouver", "Victoria Traffic", "Pacific Pilots", "Prince Rupert Coast Guard Radio", "316",
         (49, 17, "N"), (123, 6, "W"), ("12", "11"), ("Centerm", "Deltaport")),
    Port("Sydney", "Sydney VTS", "Sydney Pilots", "JRCC Australia", "503",
         (33, 50, "S"), (151, 16, "E"), ("13", "12"), ("Port Botany", "White Bay")),
    # AIS regions (see ais.py / harbor.py); receivers at or near the VTS centre
    Port("Houston", "Houston Traffic", "Houston Pilots", "Coast Guard Sector Houston Galveston", "366",
         (29, 20, "N"), (94, 44, "W"), ("12", "11"),
         ("Barbours Cut", "Bayport", "Texas City", "Bolivar Roads Anchorage"), (29.34, -94.74)),
    Port("San Francisco", "San Francisco Traffic", "San Francisco Bar Pilots", "Coast Guard Sector San Francisco",
         "366", (37, 48, "N"), (122, 22, "W"), ("14", "12"),
         ("Oakland Outer Harbor", "Richmond Long Wharf", "Anchorage Nine"), (37.81, -122.37)),
    Port("Puget Sound", "Seattle Traffic", "Puget Sound Pilots", "Coast Guard Sector Puget Sound", "366",
         (47, 36, "N"), (122, 21, "W"), ("14", "13"),
         ("Terminal Eighteen", "Elliott Bay Anchorage", "Blair Waterway"), (47.60, -122.35)),
    Port("New Orleans", "New Orleans Traffic", "Crescent River Pilots", "Coast Guard Sector New Orleans", "366",
         (29, 57, "N"), (90, 4, "W"), ("12", "63"),
         ("Napoleon Avenue Terminal", "Nashville Avenue Wharf", "Belle Chasse Anchorage"), (29.95, -90.06)),
    Port("Port Arthur", "Port Arthur Traffic", "Sabine Pilots", "Coast Guard Marine Safety Unit Port Arthur",
         "366", (29, 52, "N"), (93, 56, "W"), ("65", "13"),
         ("Sabine Pass", "Port Neches", "Beaumont Terminal"), (29.87, -93.93)),
    Port("Great Belt", "Great Belt VTS", "DanPilot", "Lyngby Radio", "219",
         (55, 21, "N"), (11, 0, "E"), ("74", "11"), ("Nyborg", "Korsor", "Kalundborg"), (55.35, 11.00)),
    Port("Oresund", "Sound VTS", "DanPilot", "Lyngby Radio", "219",
         (55, 54, "N"), (12, 42, "E"), ("73", "71"), ("Copenhagen", "Helsingborg", "Malmo"), (55.90, 12.70)),
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


# AIS extract region name (jobs/ais_extract_job.py) -> Port name
AIS_REGIONS: Dict[str, str] = {
    "new_york": "New York", "houston": "Houston", "la_long_beach": "Los Angeles",
    "san_francisco": "San Francisco", "puget_sound": "Puget Sound", "new_orleans": "New Orleans",
    "port_arthur": "Port Arthur", "great_belt": "Great Belt", "oresund": "Oresund",
}


def port_for_region(region: str) -> Port:
    name = AIS_REGIONS[region]
    return next(p for p in PORTS if p.name == name)
