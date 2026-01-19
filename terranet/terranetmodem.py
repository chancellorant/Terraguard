#!/usr/bin/env python3
"""
Telit LE910 “programming” (AT-command provisioning) script:
- sets APN (PDP context)
- attaches to packet service
- activates PDP context
- prints assigned IP

Run on Raspberry Pi where the Telit shows an AT port like /dev/ttyUSB2 or /dev/ttyUSB3.

Usage:
  sudo python3 telit_program_apn.py /dev/ttyUSB2 nxtgenphone
"""

import sys, time
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB2"
APN  = sys.argv[2] if len(sys.argv) > 2 else "nxtgenphone"

def send(ser, cmd, wait=0.4, read_timeout=2.0):
    ser.reset_input_buffer()
    ser.write((cmd + "\r").encode("ascii"))
    time.sleep(wait)

    end = time.time() + read_timeout
    buf = b""
    while time.time() < end:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            if b"\nOK" in buf or b"\nERROR" in buf:
                break
        else:
            time.sleep(0.05)

    out = buf.decode(errors="ignore").strip()
    print(f"> {cmd}\n{out}\n")
    return out

def main():
    with serial.Serial(PORT, baudrate=115200, timeout=0.2) as ser:
        # Basic comms
        send(ser, "AT")
        send(ser, "ATE0")            # echo off
        send(ser, "ATI")             # identify
        send(ser, "AT+CPIN?")        # SIM status (should be READY)
        send(ser, "AT+CREG?")        # network reg
        send(ser, "AT+CEREG?")       # LTE reg
        send(ser, "AT+CSQ")          # signal

        # Set PDP context (APN)
        # Context 1, IPv4v6 is fine for most SIMs
        send(ser, f'AT+CGDCONT=1,"IPV4V6","{APN}"')

        # Attach to packet service (may already be attached)
        send(ser, "AT+CGATT=1")

        # Activate PDP context (Telit uses #SGACT)
        # If your SIM requires user/pass, use: AT#SGACT=1,1,"user","pass"
        send(ser, "AT#SGACT=1,1")

        # Confirm active + show IP
        send(ser, "AT#SGACT?")
        send(ser, "AT+CGPADDR=1")

if __name__ == "__main__":
    main()
