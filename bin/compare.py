#!/usr/bin/python
# testing script by Casey Banner, updated by Ryan Erickson

import sys

a = open(sys.argv[1], 'r')
b = open(sys.argv[2], 'r')
max_delta = 0.0
total_error = 0.0
lines = 0

while True:
    a_line = a.readline()
    b_line = b.readline()
    
    if not a_line and not b_line:
        break

    if not a_line or not b_line:
        print ("unequal number of lines in these files")
        break

    a_pos, a_accel = a_line.split(' ')
    b_pos, b_accel = b_line.split(' ')

    split = a_pos[1:-1].split(',')
    a_pos = (split[0], split[1], split[2], split[3])
    split = b_pos[1:-1].split(',')
    b_pos = (split[0], split[1], split[2], split[3])

    if a_pos != b_pos:
        print ("position mismatch: %s vs %s" % (a_pos, b_pos) )
        break

    split = a_accel[1:-2].split(',')
    a_accel = (split[0], split[1], split[2])    
    split = b_accel[1:-2].split(',')
    b_accel = (split[0], split[1], split[2])
    
    deltas = (
        float(a_accel[0]) - float(b_accel[0]),
        float(a_accel[1]) - float(b_accel[1]),
        float(a_accel[2]) - float(b_accel[2]))

    for delta in deltas:
        if abs(delta) > max_delta:
            max_delta = abs(delta)

        total_error += abs(delta)
    

    if a_accel != b_accel:
        print ("(%s, %s, %s, %s) (%s, %s, %s)" % (a_pos[0], a_pos[1], a_pos[2], a_pos[3],
                                                 deltas[0],
                                                 deltas[1],
                                                 deltas[2]))

    lines += 1

print ("Largest delta: %f" % max_delta )
print ("Total error: %f" % total_error)
print ("Avg error / pt: %f" % (total_error / lines))

    
    
