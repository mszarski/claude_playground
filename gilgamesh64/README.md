# Gilgamesh 64

A 3D action-adventure demo in the style of late-90s N64 adventure games,
based on Tablet V of the *Epic of Gilgamesh*: the journey to the Cedar Forest
and the fight with its guardian, Humbaba.

## Status

| Track | What it is | State |
|-------|------------|-------|
| `index.html` | Browser prototype (three.js) that renders at 320x240 with 15-bit dithered colour, used to design the level and combat quickly | Playable |
| N64 ROM (`n64/`) | The same level written in C with libdragon + tiny3d, built into a `.z64` for real hardware (flashcart) and emulators | Compiles; being tested in an emulator |
| Web emulator page | The ROM running in the browser through EmulatorJS with a patched mupen64plus core | In progress |

## The demo level

1. **The Steppe of Uruk.** Start beneath the city walls with Enkidu at your side.
2. Collect the three **Seals of Shamash**:
   - *Dawn*, beside the oasis well (east).
   - *Noon*, at the scorpion-men's camp (west).
   - *Dusk*, which appears on the altar in the ring of standing stones once its lions are slain (north-east).
3. The seals open the **cedar door** to the forest.
4. **Humbaba** waits in the clearing. He slams the ground (roll through the shockwave),
   charges (dodge so he hits a tree and is stunned, which doubles your damage), and spits fire
   (raise your shield to send it back at him).

## Controls

| Action | Keyboard | Gamepad | N64 equivalent |
|--------|----------|---------|----------------|
| Move | WASD / arrows | Left stick | Control stick |
| Sword | J / X / click | X or B | B |
| Roll, talk, read | Space | A | A |
| Lock on | Shift / L / Z | LB / LT | Z |
| Shield | K / C | RB / RT | R |
| Camera | Q / E | Right stick | C-left / C-right |
| Pause | Enter / Esc | Start | Start |

V toggles a soft (VI-filter) or crisp picture, and M mutes sound. Touch controls appear on phones.

## Run it

Open `index.html` in a browser. It loads three.js from jsDelivr, so it needs a network connection.
To use a local server instead:

```bash
python3 -m http.server -d gilgamesh64 8064
```

Then visit http://localhost:8064.

## Build the N64 ROM

Needs the libdragon toolchain (preview branch) and [tiny3d](https://github.com/HailToDodongo/tiny3d), each built with its own `./build.sh`.

```bash
export N64_INST=/opt/libdragon
```

```bash
make -C gilgamesh64/n64 T3D_INST=/path/to/tiny3d
```

This produces `gilgamesh64/n64/gilgamesh64.z64`. Everything in the ROM (models, level, music) is generated in code
from `n64/src/`; the only assets are two pixel fonts (`monogram` and `abaddon`, both CC0).
