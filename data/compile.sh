#!/bin/bash
# rm -r /bin/pflotran/
# git clone https://bitbucket.org/pflotran/pflotran
diff -c /kb/module/work/tmp/scratch/reaction_sandbox_pnnl_cyber.F90 /kb/module/data/reaction_sandbox_pnnl_cyber.F90
echo "Compsrison done"
cd /bin/pflotran/src/pflotran
# make fast=1 pflotran