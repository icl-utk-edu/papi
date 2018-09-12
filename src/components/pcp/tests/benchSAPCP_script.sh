# First get a header.
PCP_NAME="perfevent.hwcounters.instructions.value"

jsrun --np 1 ./benchSAPCP 2>/dev/null  >benchSAPCP_test.csv 

# the Loop specifies how many experiments to do.
COUNTER=0
   while [  $COUNTER -lt 500 ]; do
      jsrun --np 1 ./benchSAPCP $PCP_NAME 2>/dev/null >>benchSAPCP_test.csv 
      let COUNTER=COUNTER+1 
      echo Finished: $COUNTER
   done
