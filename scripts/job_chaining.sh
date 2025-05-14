
 
 #!/bin/bash
    
FIRST=$(qsub ../run_moses_can.pbs)
echo $FIRST
SECOND=$(qsub -W depend=afterany:$FIRST ../run_guacamol_can.pbs)
echo $SECOND
THIRD=$(qsub -W depend=afterany:$SECOND ../run_moses_ClearSMILES.pbs)
echo $THIRD

