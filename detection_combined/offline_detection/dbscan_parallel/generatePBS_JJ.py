import argparse
import os
import sys
import subprocess
import shutil
from datetime import date

today = str(date.today())

parser = argparse.ArgumentParser()
parser.add_argument('--analysis', required=True, help="Which analysis should be run.")
parser.add_argument('--inputDir', required=False, default='/lcrc/group/ATLAS/users/jj/DiHydra/data/', help="Directory that contains all input DAODs.")
parser.add_argument('--outDir', required=True, help="Output directory.")
parser.add_argument('--jobFileDir', required=False, default='./', help="Directory for job files (logs, etc).")
parser.add_argument('--submit', required=False, action='store_true', help="Submit the jobs you create.")
parser.add_argument('--nNodes', required=False, type=int, default=1, help="Number of nodes to use for each job.")
parser.add_argument('--partition', required=False, default='compute',
                    choices=['bdwd', 'compute', 'bigmem', 'bigdata', 'debug'],
                    help="Partition to run on improv machine! (not bebop machine)")
parser.add_argument('--allocation', required=False, default='ATLAS-HEP-GROUP', help="Allocation to use")
parser.add_argument('--coresPerNode', required=False, type=int, default=128, help="Number of cores per node")
parser.add_argument('--jobTime', required=False, default='12:00:00', help="Time limit for jobs")
parser.add_argument('--cmd', required=False, type=str, help="Command to execute in the job script.")
args = parser.parse_args()

currentDir = os.getcwd()
jobFileDir = os.path.join(args.jobFileDir, today)
outDir = os.path.join(args.outDir, today)
os.makedirs(jobFileDir, exist_ok=True)
os.makedirs(outDir, exist_ok=True)

# PBS job script header template
jobHeader = """#!/bin/bash
#PBS -q {partition}
#PBS -N {jobName}
#PBS -l select={nodes}:ncpus={coresPerNode}
#PBS -o {jobDir}/out.log
#PBS -e {jobDir}/error.log
#PBS -l walltime={jobTime}
#PBS -A {allocation}
#PBS -j oe
"""
jobName = args.analysis.replace(".", "_")
jobFName = os.path.join(jobFileDir, f"{jobName}.job")

with open(jobFName, 'w') as job:
    job.write(jobHeader.format(partition=args.partition, jobName=jobName, nodes=args.nNodes, jobDir=jobFileDir, jobTime=args.jobTime, allocation=args.allocation, coresPerNode=args.coresPerNode))
    job.write("module load parallel\n")
    if args.cmd:
        job.write(f"echo 'Running command: {args.cmd}'\n")  #print each command to the log
        job.write(f"{args.cmd}\n")
# Submit job
if args.submit:
    print(f"Submitting job: {jobFName}")
    subprocess.call(['qsub', jobFName])
