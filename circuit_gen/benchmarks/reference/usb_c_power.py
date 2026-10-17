"""USB-C sink: 5.1k pull-downs on CC1 and CC2 request 5 V; VBUS to a header, 1k-limited power LED."""
vbus, gnd, cc1, cc2, la = Net("VBUS"), Net("GND"), Net("CC1"), Net("CC2"), Net("LED_A")
vbus.drive = gnd.drive = POWER
usb = Part("Connector", "USB_C_Receptacle_PowerOnly_6P",
           footprint="Connector_USB:USB_C_Receptacle_GCT_USB4125-xx-x_6P_TopMnt_Horizontal")
for pin in ("A9", "B9"):
    usb[pin] += vbus
for pin in ("A12", "B12", "S1"):
    usb[pin] += gnd
usb["A5"] += cc1; usb["B5"] += cc2
cc1 & R("5.1k") & gnd
cc2 & R("5.1k") & gnd
header(vbus, gnd)
vbus & C("10u") & gnd
led = Part("Device", "LED", value="green", footprint="LED_SMD:LED_0805_2012Metric")
vbus & R("1k") & la
la += led["A"]; led["K"] += gnd
