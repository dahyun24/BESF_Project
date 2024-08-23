import time
from pymodbus.server.sync import ModbusSerialServer
from pymodbus.transaction import ModbusRtuFramer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext, ModbusSlaveContext
import serial
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)



# Data Block
class ModbusSlaveDataBlock(ModbusSequentialDataBlock):
    def __init__(self):
        # Initialize with default values from address 100
        super().__init__(500, [0] * 20)  # Start address 100, 10 registers with default value 0

# Setup data store and context
data_block = ModbusSlaveDataBlock()
slave_context = ModbusSlaveContext(
    hr=data_block  # hr is for holding registers
)
store = ModbusServerContext(
    slaves=slave_context,  # single=True인 경우, ModbusSlaveContext를 바로 전달
    single=True
)


def update_modbus():
    data_block.setValues(504, [301])  # Address 203 for temperature !!one up!!
    data_block.setValues(512, [302])     # Address 212 for humidity


def run_server():
    # Define serial port settings
    port = '/dev/ttyUSB0'  # USB-to-RS485 adapter port
    baudrate = 9600
    stopbits = 1
    parity = serial.PARITY_NONE

    # Initialize Modbus server
    server = ModbusSerialServer(
        context=store,
        framer=ModbusRtuFramer,
        port=port,
        baudrate=baudrate,
        stopbits=stopbits,
        parity=parity
        )
    # Start server
    #server.serve_forever()

    # Start server in a separate thread
    import threading
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

    # Periodically update the Modbus data store
    while True:
        update_modbus()
        time.sleep(10)  # Update every 10 seconds

if __name__ == "__main__":
    run_server()
