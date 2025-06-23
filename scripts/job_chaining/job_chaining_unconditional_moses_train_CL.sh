
 
 #!/bin/bash
    
FIRST=$(qsub scripts/jobs_submitter/run_moses_ClearSMILES_CL.pbs)
echo $FIRST
SECOND=$(qsub -W depend=afterany:$FIRST scripts/jobs_submitter/run_moses_ClearSMILES_CL_longer.pbs)
echo $SECOND

