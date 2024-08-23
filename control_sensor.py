# master.py

from pymodbus.client.sync import ModbusSerialClient as ModbusClient
import serial
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

def run_client():
    # Define serial port settings
    port = '/dev/ttyUSB0'  # USB-to-RS485 adapter port
    baudrate = 9600
    stopbits = 1
    parity = serial.PARITY_NONE

    # Initialize Modbus client
    client = ModbusClient(
        method='rtu',
        port=port,
        baudrate=baudrate,
        stopbits=stopbits,
        parity=parity,
        timeout=10  # Timeout setting for response
    )
    
    try:
        client.connect()

        # Read holding registers from slave
        temp = client.read_holding_registers(202, 3, unit=1)
        #hum = client.read_holding_registers(212, 3, unit=1)
        if temp.isError(): 
            log.error(f"Error: {temp}")
        else:
            log.info(f"Read Register: {temp.registers}")

    finally:
        client.close()

if __name__ == "__main__":
    run_client()
