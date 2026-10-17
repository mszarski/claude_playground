"""BME280 in I2C mode (CSB high, SDO low for address 0x76), 4.7k pull-ups, 100n on each supply pin."""
vcc, gnd, scl, sda = Net("VCC"), Net("GND"), Net("SCL"), Net("SDA")
vcc.drive = gnd.drive = POWER
header(vcc, gnd, scl, sda)
s = Part("Sensor", "BME280", footprint="Package_LGA:Bosch_LGA-8_2.5x2.5mm_P0.65mm_ClockwisePinNumbering")
s["VDD"] += vcc; s["VDDIO"] += vcc; s[1] += gnd; s[7] += gnd
s["CSB"] += vcc; s["SDO"] += gnd
s["SDI"] += sda; s["SCK"] += scl
vcc & R("4.7k") & sda
vcc & R("4.7k") & scl
vcc & C("100n") & gnd
vcc & C("100n") & gnd
