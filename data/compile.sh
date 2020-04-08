#!/bin/bash
rm -r /bin/pflotran/
git clone https://bitbucket.org/pflotran/pflotran
# cp /kb/module/work/tmp/scratch/reaction_sandbox_pnnl_cyber.F90 /bin/pflotran/src/pflotran
cd /bin/pflotran/src/pflotran
make pflotran