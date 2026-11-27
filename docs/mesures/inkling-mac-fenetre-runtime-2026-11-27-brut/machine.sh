#!/bin/bash
# One line of machine state: load at one minute, GPU utilisation, free memory, and pmset's thermal notes (the CPU
# speed limit when one is recorded), joined on one line.
printf 'load1 %s gpu %s%% memfree %s therm [%s]\n' "$(sysctl -n vm.loadavg | awk '{print $2}')" \
  "$(ioreg -r -d 1 -w 0 -c IOAccelerator | sed -n 's/.*"Device Utilization %"=\([0-9]*\).*/\1/p' | head -1)" \
  "$(memory_pressure -Q | awk -F': ' '/percentage/{print $2}')" \
  "$(pmset -g therm 2>&1 | tr '\n' ';' | sed 's/; *$//')"
