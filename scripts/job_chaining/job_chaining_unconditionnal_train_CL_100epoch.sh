
 
 #!/bin/bash
    
FIRST=$(qsub scripts/clearsmi_moses_CL_corrected/run_100epoch.pbs)
echo $FIRST
SECOND=$(qsub -W depend=afterany:$FIRST scripts/cansmi_moses_CL_corrected/run_100epoch.pbs)
echo $SECOND


