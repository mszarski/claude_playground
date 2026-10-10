// ---------- Game data: species, buildings, constants ----------
'use strict';

const TILE = 16;
const MAP_W = 72;
const MAP_H = 54;

// Terrain ids
const T_DEEP = 0, T_WATER = 1, T_SAND = 2, T_GRASS = 3, T_FOREST = 4, T_ROCK = 5, T_LAVA = 6, T_BASALT = 7, T_VOLCANO = 8;

// Fence ids (stored per tile)
const F_NONE = 0, F_ELECTRIC = 1, F_WALL = 2, F_BROKEN = 3;

const FENCE_DEF = {
  [F_ELECTRIC]: { name: 'Electric Fence', hp: 100, cost: 250, power: 0.5 },
  [F_WALL]: { name: 'Concrete Wall', hp: 420, cost: 900, power: 0 },
};

// Dinosaur species
// strength: fence damage per hit; danger: threat to humans (0 = harmless)
// space: tiles needed each; social: preferred minimum group; unlock: star rating needed
const SPECIES = {
  galli: {
    name: 'Gallimimus', sci: 'Gallimimus bullatus', diet: 'herb', cost: 22000, appeal: 3, strength: 1, danger: 1,
    speed: 3.4, hp: 60, tranq: 1, space: 14, social: 3, forest: 0.15, unlock: 0, size: 1,
    colors: { O: '#2a1d14', B: '#b88a4a', D: '#8a5e32', L: '#e8d0a0', S: '#7a4a2a', E: '#141018' },
    desc: 'Fast, skittish flock runner. Happiest in large groups.'
  },
  para: {
    name: 'Parasaurolophus', sci: 'Parasaurolophus walkeri', diet: 'herb', cost: 34000, appeal: 4, strength: 2, danger: 0,
    speed: 2.2, hp: 90, tranq: 1, space: 18, social: 2, forest: 0.3, unlock: 0, size: 2,
    colors: { O: '#1e1a12', B: '#5f8a4a', D: '#3f6232', L: '#d8d0a0', S: '#c85a2a', E: '#141018' },
    desc: 'Gentle crested herbivore. Hooting herds delight guests.'
  },
  dilo: {
    name: 'Dilophosaurus', sci: 'Dilophosaurus wetherilli', diet: 'carn', cost: 45000, appeal: 5, strength: 2, danger: 4,
    speed: 3.0, hp: 70, tranq: 1, space: 16, social: 1, forest: 0.4, unlock: 0, size: 1,
    colors: { O: '#14181a', B: '#7a9a52', D: '#4a6a3a', L: '#e0d890', S: '#e86a2a', E: '#f8e040' },
    desc: 'Frilled venom-spitter. Small, but nasty when loose.'
  },
  trike: {
    name: 'Triceratops', sci: 'Triceratops horridus', diet: 'herb', cost: 60000, appeal: 6, strength: 6, danger: 2,
    speed: 1.5, hp: 180, tranq: 2, space: 30, social: 1, forest: 0.2, unlock: 1, size: 2,
    colors: { O: '#1a1410', B: '#8a6a52', D: '#5e4636', L: '#c8b090', S: '#b84a2a', E: '#141018' },
    desc: 'Armored and stubborn. Charges fences when stressed.'
  },
  stego: {
    name: 'Stegosaurus', sci: 'Stegosaurus stenops', diet: 'herb', cost: 52000, appeal: 5, strength: 4, danger: 1,
    speed: 1.3, hp: 160, tranq: 2, space: 26, social: 2, forest: 0.3, unlock: 1, size: 2,
    colors: { O: '#18160e', B: '#6a7a4a', D: '#4a5632', L: '#c8c890', S: '#d07a2a', E: '#141018' },
    desc: 'Plated giant with a spiked tail. Mostly peaceful.'
  },
  raptor: {
    name: 'Velociraptor', sci: 'Velociraptor antirrhopus', diet: 'carn', cost: 95000, appeal: 8, strength: 3, danger: 8,
    speed: 4.4, hp: 90, tranq: 2, space: 20, social: 3, forest: 0.35, unlock: 2, size: 1, clever: true,
    colors: { O: '#16120e', B: '#a0784a', D: '#6a4a2a', L: '#e0c898', S: '#4a3020', E: '#f8d040' },
    desc: 'Pack hunter. Systematically tests fences for weaknesses.'
  },
  anky: {
    name: 'Ankylosaurus', sci: 'Ankylosaurus magniventris', diet: 'herb', cost: 72000, appeal: 5, strength: 7, danger: 2,
    speed: 1.1, hp: 240, tranq: 3, space: 26, social: 1, forest: 0.2, unlock: 2, size: 2,
    colors: { O: '#14120e', B: '#7a6a4a', D: '#4e4232', L: '#b8a880', S: '#3e3428', E: '#141018' },
    desc: 'Living tank with a club tail. Fences beware.'
  },
  brachio: {
    name: 'Brachiosaurus', sci: 'Brachiosaurus altithorax', diet: 'herb', cost: 130000, appeal: 10, strength: 8, danger: 1,
    speed: 0.9, hp: 400, tranq: 4, space: 50, social: 2, forest: 0.45, unlock: 3, size: 3,
    colors: { O: '#141a18', B: '#6a8a7a', D: '#4a6258', L: '#b8c8a8', S: '#587868', E: '#141018' },
    desc: 'The welcome-to-the-park moment. Guests adore it.'
  },
  trex: {
    name: 'Tyrannosaurus', sci: 'Tyrannosaurus rex', diet: 'carn', cost: 220000, appeal: 14, strength: 10, danger: 10,
    speed: 2.6, hp: 380, tranq: 5, space: 60, social: 1, forest: 0.25, unlock: 3, size: 3, territorial: true,
    colors: { O: '#140e0a', B: '#7a5a3e', D: '#4e3624', L: '#b89a72', S: '#5a3a24', E: '#f0a020' },
    desc: 'The main attraction. Do not let her out. Ever.'
  },
  spino: {
    name: 'Spinosaurus', sci: 'Spinosaurus aegyptiacus', diet: 'carn', cost: 280000, appeal: 15, strength: 10, danger: 10,
    speed: 2.4, hp: 420, tranq: 6, space: 60, social: 1, forest: 0.2, unlock: 4, size: 3, territorial: true,
    colors: { O: '#120e0e', B: '#6a5a5e', D: '#443a3e', L: '#b8a8a0', S: '#c84a3a', E: '#f8c020' },
    desc: 'Sail-backed apex predator. Needs water in its paddock.'
  },
};
const SPECIES_ORDER = ['galli', 'para', 'dilo', 'trike', 'stego', 'raptor', 'anky', 'brachio', 'trex', 'spino'];

// Buildings. w/h are tile footprint.
// cat: 'guest' (needs path access), 'infra', 'dino', 'staff', 'decor'
const BUILDINGS = {
  gate: { name: 'Main Gate', w: 3, h: 2, cost: 15000, upkeep: 100, power: 0, cat: 'guest', hp: 300, unique: true,
    desc: 'Guests enter and exit here. Must touch a path.' },
  visitor: { name: 'Visitor Center', w: 3, h: 3, cost: 60000, upkeep: 400, power: 6, cat: 'guest', hp: 400, shelter: 60, income: 6, appeal: 8,
    desc: 'Guest hub and emergency shelter for 60 guests.' },
  restaurant: { name: 'Restaurant', w: 2, h: 2, cost: 18000, upkeep: 200, power: 3, cat: 'guest', hp: 200, income: 22, need: 'hunger',
    desc: 'Feeds hungry guests. Good income.' },
  shop: { name: 'Gift Shop', w: 2, h: 2, cost: 14000, upkeep: 120, power: 2, cat: 'guest', hp: 160, income: 18, need: 'shop',
    desc: 'Plush raptors and amber keychains.' },
  restroom: { name: 'Restroom', w: 1, h: 1, cost: 4000, upkeep: 40, power: 1, cat: 'guest', hp: 100, income: 0, need: 'toilet',
    desc: 'Guests get grumpy without these. Not a great hiding place.' },
  viewing: { name: 'Viewing Platform', w: 2, h: 2, cost: 9000, upkeep: 40, power: 0, cat: 'guest', hp: 220, appeal: 3, view: 11,
    desc: 'Place by a paddock. Guests here see dinos up to 11 tiles away.' },
  hotel: { name: 'Lodge Hotel', w: 3, h: 3, cost: 90000, upkeep: 600, power: 8, cat: 'guest', hp: 400, income: 140, shelter: 50, need: 'rest', appeal: 4,
    desc: 'Overnight stays. Big income and a shelter for 50.' },
  shelter: { name: 'Bunker', w: 2, h: 2, cost: 12000, upkeep: 60, power: 1, cat: 'guest', hp: 900, shelter: 45,
    desc: 'Reinforced emergency bunker for 45 guests.' },
  power: { name: 'Power Plant', w: 3, h: 3, cost: 45000, upkeep: 500, power: 0, supply: 110, cat: 'infra', hp: 350, range: 13,
    desc: 'Generates 110 MW and powers a radius of 13 tiles.' },
  pylon: { name: 'Power Pylon', w: 1, h: 1, cost: 1200, upkeep: 10, power: 0, cat: 'infra', hp: 120, range: 8,
    desc: 'Extends grid coverage by 8 tiles. Must be inside existing coverage.' },
  backup: { name: 'Backup Generator', w: 2, h: 2, cost: 28000, upkeep: 150, power: 0, backup: 70, cat: 'infra', hp: 260, range: 8,
    desc: 'Kicks in during grid failure: 70 MW for a while.' },
  ranger: { name: 'Ranger Station', w: 2, h: 2, cost: 24000, upkeep: 450, power: 2, cat: 'staff', hp: 260, staff: 'ranger', staffN: 2,
    desc: 'Two rangers with tranquilizer rifles.' },
  maint: { name: 'Maintenance Shed', w: 2, h: 2, cost: 14000, upkeep: 300, power: 1, cat: 'staff', hp: 220, staff: 'worker', staffN: 2,
    desc: 'Two engineers who repair fences and buildings.' },
  helipad: { name: 'ACU Helipad', w: 3, h: 3, cost: 65000, upkeep: 700, power: 2, cat: 'staff', hp: 300,
    desc: 'Asset Containment helicopter airlifts sedated dinos home.' },
  vet: { name: 'Vet Clinic', w: 2, h: 2, cost: 30000, upkeep: 350, power: 2, cat: 'staff', hp: 220,
    desc: 'Treats sick dinosaurs and stops outbreaks.' },
  hatchery: { name: 'Hatchery', w: 3, h: 2, cost: 40000, upkeep: 300, power: 4, cat: 'dino', hp: 300, unique: true,
    desc: 'Required to hatch dinosaurs.' },
  feeder_h: { name: 'Herbivore Feeder', w: 1, h: 1, cost: 2500, upkeep: 20, power: 0, cat: 'dino', hp: 120, feeds: 'herb',
    desc: 'Place inside a paddock. Each meal costs $150.' },
  feeder_c: { name: 'Carnivore Feeder', w: 1, h: 1, cost: 4500, upkeep: 30, power: 0, cat: 'dino', hp: 120, feeds: 'carn',
    desc: 'Live goats. Place inside a paddock. Each meal costs $400.' },
  lamp: { name: 'Lamp Post', w: 1, h: 1, cost: 300, upkeep: 2, power: 0.2, cat: 'decor', hp: 60, light: true,
    desc: 'Lights the night. Guests feel safer.' },
  tour: { name: 'Tour Station', w: 2, h: 2, cost: 35000, upkeep: 300, power: 3, cat: 'guest', hp: 250, income: 25, appeal: 6, need: 'tour',
    desc: 'Electric jeeps carry guests along Tour Track past the paddocks. Must touch a path and a track.' },
  siren: { name: 'Siren Tower', w: 1, h: 1, cost: 6000, upkeep: 20, power: 1, cat: 'infra', hp: 120, range: 14,
    desc: 'Guests in range evacuate instantly when an alarm sounds.' },
};

const TOOL_GROUPS = [
  { id: 'inspect', label: 'Inspect', tools: ['inspect'] },
  { id: 'build', label: 'Build', tools: ['path', 'track', 'fence', 'paddock', 'wall', 'demolish', 'trees', 'clear'] },
  { id: 'guest', label: 'Guests', tools: ['gate', 'visitor', 'restaurant', 'shop', 'restroom', 'viewing', 'tour', 'hotel', 'shelter', 'lamp'] },
  { id: 'infra', label: 'Power', tools: ['power', 'pylon', 'backup', 'siren'] },
  { id: 'staff', label: 'Staff', tools: ['ranger', 'maint', 'helipad', 'vet'] },
  { id: 'dino', label: 'Dinos', tools: ['hatchery', 'feeder_h', 'feeder_c', 'hatch'] },
];

const TOOL_INFO = {
  inspect: { name: 'Inspect', desc: 'Click anything to see details.', key: 'Q' },
  path: { name: 'Footpath', cost: 60, desc: 'Drag to lay paths. Guests only walk on paths.', key: 'P' },
  track: { name: 'Tour Track', cost: 150, desc: 'Drag to lay the jeep tour track. Run it past paddocks and loop it back to a Tour Station.', key: 'J' },
  fence: { name: 'Electric Fence', cost: 250, desc: 'Drag a line of electric fence. Needs power!', key: 'F' },
  paddock: { name: 'Paddock', cost: 250, desc: 'Drag a rectangle to build a fenced paddock.', key: 'R' },
  wall: { name: 'Concrete Wall', cost: 900, desc: 'Drag. Very strong, needs no power.', key: 'V' },
  demolish: { name: 'Demolish', cost: 0, desc: 'Drag to remove paths, fences and buildings.', key: 'X' },
  trees: { name: 'Plant Trees', cost: 80, desc: 'Drag to plant jungle. Dinos love cover.', key: 'T' },
  clear: { name: 'Clear Land', cost: 40, desc: 'Drag to clear jungle.', key: 'C' },
  hatch: { name: 'Hatch Dinosaur', desc: 'Pick a species, then click inside a paddock.', key: 'H' },
};
