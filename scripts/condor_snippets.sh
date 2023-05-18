# List of useful condor commands:
# This file is not meant to be run directly

# Summit
condor_submit ./chtc/job.sub

# Query the status of the job
condor_q

# If it haven't start for a bit, better check:
condor_q -better-analyze <job_id>

# Remove job
condor_rm <job_id>

# Check the number of GPUs available on the CHTC
condor_status -compact -constraint 'TotalGpus > 0'
