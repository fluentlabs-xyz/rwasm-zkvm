#!/bin/bash

cargo +nightly fmt --all 

cargo +stable clippy -p rwasm-machine -p rwasm-executor -p rwasm-machine-test --fix --allow-staged