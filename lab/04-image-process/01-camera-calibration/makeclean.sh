#!/bin/bash
function main() {
    set -e     # stop script and exit the terminal when got any error
    set -x 

    rm -fr	./results/*
    rm -f *.pkl    

    set +e
    set +x 
    echo "Done!"
}

main
