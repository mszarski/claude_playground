"""5 V to 3.3 V linear regulator board with a power LED."""

vin, v33, gnd = Net("VIN"), Net("+3V3"), Net("GND")
vin.drive = gnd.drive = POWER  # supplied through the input connector

j_in = Part("Connector", "Conn_01x02_Pin",
            footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical")
j_out = Part("Connector", "Conn_01x02_Pin",
             footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical")
reg = Part("Regulator_Linear", "AMS1117-3.3", footprint="Package_TO_SOT_SMD:SOT-223-3_TabPin2")
c_in = Part("Device", "C", value="10u", footprint="Capacitor_SMD:C_0805_2012Metric")
c_out = Part("Device", "C", value="22u", footprint="Capacitor_SMD:C_0805_2012Metric")
led = Part("Device", "LED", value="green", footprint="LED_SMD:LED_0805_2012Metric")
r_led = Part("Device", "R", value="1k", footprint="Resistor_SMD:R_0805_2012Metric")

j_in[1, 2] += vin, gnd
j_out[1, 2] += v33, gnd
reg["VI"] += vin
reg["VO"] += v33
reg["GND"] += gnd
c_in[1, 2] += vin, gnd
c_out[1, 2] += v33, gnd
v33 & r_led & led["A K"] & gnd

# Requirements, checked in simulation across component tolerances.
require("v(+3V3)", 3.2, 3.4, sources={"VIN": 5}, loads={"+3V3": 33},
        name="3.3 V at 100 mA")
require("i(VIN)", 0.0005, 0.003, sources={"VIN": 5}, name="no-load input current (LED)")
