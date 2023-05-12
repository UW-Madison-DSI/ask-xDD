#!/bin/bash

# Check the number of GPUs available on the CHTC
condor_status -compact -constraint 'TotalGpus > 0'