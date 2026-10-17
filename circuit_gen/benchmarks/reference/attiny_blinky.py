"""ATtiny85 with 100n decoupling, 10k reset pull-up, 6-pin ISP header (MOSI=PB0, MISO=PB1,
SCK=PB2, RST=PB5) and an LED on PB0 through 330 R."""
vcc, gnd, rst, pb0, pb1, pb2, la = (Net("VCC"), Net("GND"), Net("RESET"), Net("PB0"),
                                    Net("PB1"), Net("PB2"), Net("LED_A"))
vcc.drive = gnd.drive = POWER
header(vcc, gnd)
u = Part("MCU_Microchip_ATtiny", "ATtiny85-20S", footprint="Package_SO:SOIC-8W_5.3x5.3mm_P1.27mm")
u["VCC"] += vcc; u["GND"] += gnd
u[1] += rst; u[5] += pb0; u[6] += pb1; u[7] += pb2
isp = Part("Connector", "AVR-ISP-6", footprint="Connector_PinHeader_2.54mm:PinHeader_2x03_P2.54mm_Vertical")
isp["MISO"] += pb1; isp["VCC"] += vcc; isp["SCK"] += pb2
isp["MOSI"] += pb0; isp[5] += rst; isp["GND"] += gnd
vcc & R("10k") & rst
vcc & C("100n") & gnd
led = Part("Device", "LED", value="red", footprint="LED_SMD:LED_0805_2012Metric")
pb0 & R("330") & la
la += led["A"]; led["K"] += gnd
