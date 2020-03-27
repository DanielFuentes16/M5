#!/bin/bash

FILE=$1

cat $FILE | grep d2.utils.events | awk '{print $8 " " $10}' > loss.txt
