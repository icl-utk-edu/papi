# -----------------------------------------------------------------------------
# This script will make a newpmid.txt. Provide as a first argument any 
# text needed to execute an executable on the target system. Example:
# sh newmpid.sh "jsrun --np 1"
# It excludes 'perfevent.' lines; they are included automatically. 

# Construct a file of all PCP events available.
$1 pminfo -t | sort >temppmid.txt

# Use the stream editor to delete any lines starting with 'perfevent.'
# To prevent user confusion; they cannot exclude any perfevents. 
sed -i -e "/^perfevent\./d" temppmid.txt

# Use the stream editor to add '# ' to every line.
sed -i -e "s/^/\# /" temppmid.txt

# Output the file header.
echo "# The '#' character in the first column is a comment line.            "     >newpmid.txt
echo "# This file was constructed by the script 'newpmid.sh'.               "    >>newpmid.txt
echo "# The space ' ' character indicates an event you wish to be included. "    >>newpmid.txt
echo "# You can align the event names; any leading spaces will be discarded."    >>newpmid.txt
echo "#                                                                     "    >>newpmid.txt
echo "# The program 'buildEventList' reads this file and constructs the     "    >>newpmid.txt
echo "# C-code EventList.c to be included in linux-pcp.c, so the PCP events "    >>newpmid.txt
echo "# are known at initialization time.                                   "    >>newpmid.txt
cat temppmid.txt                                                                 >>newpmid.txt
rm temppmid.txt

