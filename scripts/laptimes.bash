#!/bin/bash

kubectl get pods | grep -Eo "^worker-pods-[[:alnum:]]+" | while read -r pod ; do
	kubectl logs pods/$pod -c worker-container | grep -E "^Lap times: \[[4-7]+\.[0-9]+," | while read -r line; do
        echo $pod $line
   done
   sleep .25
done
