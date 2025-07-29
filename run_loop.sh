FIRST=$(qsub run.pbs)
echo $FIRST
SECOND=$(qsub -W depend=afterany:$FIRST run2.pbs)
echo $SECOND