
 
 #!/bin/bash
    
FIRST=$(qsub scripts/jobs_submitter/run_moses_can_BEP.pbs)
echo $FIRST
SECOND=$(qsub -W depend=afterany:$FIRST scripts/jobs_submitter/run_guacamol_can_BEP.pbs)
echo $SECOND
THIRD=$(qsub -W depend=afterany:$SECOND scripts/jobs_submitter/run_moses_ClearSMILES_BEP.pbs)
echo $THIRD

