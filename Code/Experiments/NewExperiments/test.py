from smbus2 import SMBus, i2c_msg

DLP300_ADDR = 0x1B
bus = SMBus(1)
def write_reg(reg,value):
    data= value.to_bytes(4,byteorder='big'
                         )
    msg = i2c_msg.write(DLP300_ADDR,[reg]+list(data))
    bus.i2c_rdwr(msg)

write_reg(0x16,0x00000001)
write_reg(0x13,0x00000000)
write_reg(0x14,0x00000000)