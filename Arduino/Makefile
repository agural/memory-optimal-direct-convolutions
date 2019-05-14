# Makefile

NO_CORE					= Yes

MCU 					= atmega328p
F_CPU 					= 16000000L

# https://www.avrfreaks.net/forum/programming-arduino-nano-avrdude
AVRDUDE_ARD_PROGRAMMER	= arduino
AVRDUDE_ARD_BAUDRATE	= 57600
MONITOR_PORT			= /dev/ttyUSB*

AVRDUDE					= /usr/bin/avrdude
AVRDUDE_CONF			= /etc/avrdude.conf

AVR_TOOLS_DIR			= /usr

#CXXFLAGS				+= -c -g -O1 -w -std=gnu++11 -fpermissive -fno-exceptions -ffunction-sections -fdata-sections -fno-threadsafe-statics -finline-functions -ffreestanding -flto -fwhole-program -Wl,--gc-sections -Wl,--relax -w -x c++ -CC #-E
# https://p5r.uk/blog/2008/avr-gcc-optimisations.html
CXXFLAGS				+= -c -g -O1 -w -std=gnu++11 -fpermissive -fno-exceptions -ffunction-sections -fdata-sections -fno-threadsafe-statics -finline-functions -ffreestanding -fstack-reuse=all -fstack-usage -fverbose-asm -flto -Wl,--gc-sections -w -x c++ -CC #-E

include ./Arduino-Makefile/Arduino.mk

