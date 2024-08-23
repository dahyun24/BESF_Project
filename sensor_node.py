import time
import adafruit_dht
import board
import RPi.GPIO as GPIO
from pymodbus.server.sync import ModbusSerialServer
from pymodbus.transaction import ModbusRtuFramer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext, ModbusSlaveContext
import serial
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

# Setup GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)

# DHT11 센서 초기화
dht_sensor = adafruit_dht.DHT11(board.D17)

# Data Block
class ModbusSlaveDataBlock(ModbusSequentialDataBlock):
    def __init__(self):
        # Initialize with default values from address 100
        super().__init__(200, [0] * 20)  # Start address 100, 10 registers with default value 0

# Setup data store and context
data_block = ModbusSlaveDataBlock()
slave_context = ModbusSlaveContext(
    hr=data_block  # hr is for holding registers
)
store = ModbusServerContext(
    slaves=slave_context,  # single=True인 경우, ModbusSlaveContext를 바로 전달
    single=True
)

def read_sensor():
    """Read temperature and humidity from the DHT11 sensor."""
    try:
        print("8")
        temperature = dht_sensor.temperature
        humidity = dht_sensor.humidity
        print(f"temp = {temperature}, humi = {humidity}")

        if temperature is None or humidity is None:
            raise ValueError("Sensor returned invalid data")

        # Convert temperature and humidity to Modbus format
        temperature = int(temperature * 10)  # Convert to integer (e.g., 25.3°C becomes 253)
        humidity = int(humidity * 10)        # Convert to integer (e.g., 55.2% becomes 552)

        return temperature, humidity
    except Exception as e:
        log.error(f"Error reading sensor data: {e}")
        return None, None

def update_modbus():
    print("6")
    """Update Modbus data store with the latest sensor readings."""
    temperature, humidity = read_sensor()
    print("7")
    if temperature is not None and humidity is not None:
        # Update Modbus data block
        data_block.setValues(203, [temperature])  # Address 203 for temperature
        data_block.setValues(212, [humidity])     # Address 212 for humidity


def run_server():
    # Define serial port settings
    port = '/dev/ttyUSB0'  # USB-to-RS485 adapter port
    baudrate = 9600
    stopbits = 1
    parity = serial.PARITY_NONE
    print("1")

    # Initialize Modbus server
    server = ModbusSerialServer(
        context=store,
        framer=ModbusRtuFramer,
        port=port,
        baudrate=baudrate,
        stopbits=stopbits,
        parity=parity
        )
    print("2")
    # Start server
    #server.serve_forever()

    # Start server in a separate thread
    import threading
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    print("3")

    # Periodically update the Modbus data store
    while True:
        print("4")
        update_modbus()
        print("5")
        time.sleep(10)  # Update every 10 seconds

if __name__ == "__main__":
    run_server()
