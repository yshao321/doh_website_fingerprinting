#!/bin/bash
# 
# This is a script to collect DNS-over-HTTPS traffic
# 
# Configuration
# - Web browser : Firefox with DoH enabled
# - DoH client  : N/A
# - DoH server  : mozilla.cloudflare-dns.com
# 

# Firefox and tcpdump time in seconds => about 1 minute for one website
FIREFOX_STANDBY=10
FIREFOX_SURFING=30
FIREFOX_CLOSING=10
TCPDUMP_STANDBY=5
TCPDUMP_CLOSING=5

# Tcpdump interface
TCPDUMP_INTERFACE='ens160'

# Tcpdump expression
TCPDUMP_EXPRESSION='(host 104.16.248.249 or host 104.16.249.249) and (port 443)'

# Website starting id and stopping id and number of samples
WEBSITE_STARTING_ID=1
WEBSITE_STOPPING_ID=1
WEBSITE_SAMPLES_NUM=1

# Exit due to abnormal input
exit_abnormal_input() {
    echo "Usage: $0 [ -a WEBSITE_STARTING_ID ] [ -z WEBSITE_STOPPING_ID ] [ -n WEBSITE_SAMPLES_NUM ]" 1>&2
  exit 1
}

# Read options from command line arguments
while getopts ":a:z:n:" options; do
  case "${options}" in
    a)
      WEBSITE_STARTING_ID=${OPTARG}
      if ! [[ $WEBSITE_STARTING_ID =~ ^[0-9]+$ ]]; then
        echo "Error: WEBSITE_STARTING_ID must be a positive, whole number." 1>&2
        exit_abnormal_input
      elif [ $WEBSITE_STARTING_ID -eq 0 ]; then
        echo "Error: WEBSITE_STARTING_ID must be greater than zero." 1>&2
        exit_abnormal_input
      fi
      ;;
    z)
      WEBSITE_STOPPING_ID=${OPTARG}
      if ! [[ $WEBSITE_STOPPING_ID =~ ^[0-9]+$ ]]; then
        echo "Error: WEBSITE_STOPPING_ID must be a positive, whole number." 1>&2
        exit_abnormal_input
      elif [ $WEBSITE_STOPPING_ID -eq 0 ]; then
        echo "Error: WEBSITE_STOPPING_ID must be greater than zero." 1>&2
        exit_abnormal_input
      fi
      ;;
    n)
      WEBSITE_SAMPLES_NUM=${OPTARG}
      if ! [[ $WEBSITE_SAMPLES_NUM =~ ^[0-9]+$ ]]; then
        echo "Error: WEBSITE_SAMPLES_NUM must be a positive, whole number." 1>&2
        exit_abnormal_input
      elif [ $WEBSITE_SAMPLES_NUM -eq 0 ]; then
        echo "Error: WEBSITE_SAMPLES_NUM must be greater than zero." 1>&2
        exit_abnormal_input
      fi
      ;;
    :)
      echo "Error: -${OPTARG} requires an argument." 1>&2
      exit_abnormal_input
      ;;
    *)
      exit_abnormal_input
      ;;
  esac
done

# Websites CSV file has two columns: website id (ascending) and website name
WEBSITES_CSV='../collection/top-1m.csv'
[ ! -f $WEBSITES_CSV ] && { echo "$WEBSITES_CSV file not found" 1>&2; exit 2; }

# Enter sudo password before collection
read -sp 'Enter sudo password to start collection: ' PASSWORD

# If a pipe exists on stdout, it is for real-time prediction
if [ -p /dev/stdout ]; then
  echo $PASSWORD | sudo -S tcpdump -l -i $TCPDUMP_INTERFACE $TCPDUMP_EXPRESSION --immediate-mode
  exit 0
fi

################################################################################################
echo a=$WEBSITE_STARTING_ID z=$WEBSITE_STOPPING_ID n=$WEBSITE_SAMPLES_NUM

# Prevent firefox showing the safe mode dialog after crash
export MOZ_DISABLE_AUTO_SAFE_MODE=1

mkdir -p ../collection

for ((i=1; i<=WEBSITE_SAMPLES_NUM; i++))
do
  # Create a folder for each cycle
  current_date_time="`date +%Y%m%d%H%M%S`"
  current_date_time="${WEBSITE_STARTING_ID}-${WEBSITE_STOPPING_ID}-${current_date_time}"
  mkdir -p ../collection/$current_date_time

  # One cycle of data collection
  while IFS=, read website_id website_name
  do
    # If website_id or website_name is empty, skip this website
    if [ "$website_id" == "" ] || [ "$website_name" == "" ]; then
      continue
    fi
    # If website_id is not a positive integer, skip this website
    if ! [ $website_id -gt 0 ]; then
      continue
    fi
    # If website_id is less than website_starting_id, skip this website
    if [ $WEBSITE_STARTING_ID -gt 0 ] && [ $website_id -lt $WEBSITE_STARTING_ID ]; then
      continue
    fi
    # If website_id is greater than website_stopping_id, stop this cycle
    if [ $WEBSITE_STOPPING_ID -gt 0 ] && [ $website_id -gt $WEBSITE_STOPPING_ID ]; then
      break
    fi

    echo "Collecting:" a=$WEBSITE_STARTING_ID z=$WEBSITE_STOPPING_ID n=$WEBSITE_SAMPLES_NUM i=$i $website_id $website_name

    # Start firefox
    firefox &
    sleep $FIREFOX_STANDBY

    # Start traffic collection
    echo $PASSWORD | sudo -S tcpdump -U -i $TCPDUMP_INTERFACE $TCPDUMP_EXPRESSION -w ../collection/$current_date_time/$website_id.pcap &
    sleep $TCPDUMP_STANDBY

    # Open a tab in firefox
    firefox --new-tab $website_name & 
    sleep $FIREFOX_SURFING

    # Stop traffic collection
    echo $PASSWORD | sudo -S pkill -f tcpdump
    sleep $TCPDUMP_CLOSING

    # Close firefox (clean DNS cache)
    pkill -f firefox
    sleep $FIREFOX_CLOSING

    # Clean cache file, browsing histroy, session store ...
    rm -rf ~/.cache/mozilla/firefox/*
    rm -rf ~/.mozilla/firefox/*.default*/*.sqlite
    rm -rf ~/.mozilla/firefox/*.default*/sessionstore*
  done < $WEBSITES_CSV
done

