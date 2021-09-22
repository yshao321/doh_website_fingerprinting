#!/bin/bash
# 
# This is a script to sanitize websites
# 

USER_AGENT=' -H "User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0 Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"'
ACCEPT=' -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"'
ACCEPT_LANGUAGE=' -H "Accept-Language: en-US,en;q=0.5"'
ACCEPT_ENCODING=' -H "Accept-Encoding: gzip, deflate"'
CONNECTION=' -H "Connection: keep-alive"'
UPGRADE_INSECURE_REQUEST=' -H "Upgrade-Insecure-Requests: 1"'

CURL_HEADERS=${USER_AGENT}
CURL_HEADERS+=${ACCEPT}
CURL_HEADERS+=${ACCEPT_LANGUAGE}
CURL_HEADERS+=${ACCEPT_ENCODING}
CURL_HEADERS+=${CONNECTION}
CURL_HEADERS+=${UPGRADE_INSECURE_REQUEST}

website_hpcode="000"
website_return="none"
website_status=0

function is_website_black {
  curl_command='curl -sL -w "%{http_code}" "$1" -o website_file --max-time 10'
  curl_command+=' --retry-delay 1 --retry 2'
  curl_command+=${CURL_HEADERS}
  
  ## Update website http code
  website_hpcode=$(eval "$curl_command")
  website_return="none"
  website_status=0
  
  ## Update website return
  if [ -f website_file ]; then
    website_return="okay"
    
    ### check website size
    website_size=`stat -c%s website_file`
    if (( website_size < 20 )); then
      website_return="empty"
    fi
    
    ### check website keywords
    if grep   -iq "Web Page Blocked" website_file; then
      website_return="blocked"
    elif grep -iq "Not Found"        website_file; then
      website_return="not_found"
    elif grep -iq "403 Forbidden"    website_file; then
      website_return="forbidden"
    elif grep -iq "Access Denied"    website_file; then
      website_return="denied"
    elif grep -iq "Invalid URL"      website_file; then
      website_return="invalid"
    elif grep -iq "<Error>"          website_file; then
      website_return="error"
    fi
    
    rm website_file
  fi
  
  printf '%-10s %-10s\n' $website_hpcode $website_return
  
  ## Update website status
  case $website_hpcode in
  [2]*)
    if [[ $website_return != "okay" ]]; then
      website_status=2
    fi
    ;;
  [3]*)
    ### Treat URL redirection as success
    ;;
  [4]*)
    if [[ $website_return != "okay" ]]; then
      website_status=4
    fi
    ;;
  [5]*)
    if [[ $website_return != "okay" ]]; then
      website_status=5
    fi
    ;;
  *)
    website_status=6
    ;;
  esac
}

# Website input file
WEBSITES_INPUT='../collection/top-1m.csv'

# Exit due to abnormal input
exit_abnormal_input() {
    echo "Usage: $0 [ -f WEBSITE_INPUT_FILE ]" 1>&2
  exit 1
}

# Read options from command line arguments
while getopts ":f:" options; do
  case "${options}" in
    f)
      WEBSITES_INPUT=${OPTARG}
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

# Websites input file has two columns: website id and website name
[ ! -f $WEBSITES_INPUT ] && { echo "$WEBSITES_INPUT file not found" 1>&2; exit 2; }

# Website output file
WEBSITES_BLACK="${WEBSITES_INPUT}.blk"

# Websites output file has three columns: website id, website name, and website return
[ -f $WEBSITES_BLACK ] && { echo "$WEBSITES_BLACK file already exists" 1>&2; exit 3; }

################################################################################################

while IFS=, read website_id website_name website_xxx website_yyy
do
    # If website_id or website_name is empty, skip this website
    if [ "$website_id" == "" ] || [ "$website_name" == "" ]; then
      continue
    fi
    # If website_id is not a positive integer, skip this website
    if ! [ $website_id -gt 0 ]; then
      continue
    fi

    printf 'Connecting: %-10s %-50s' $website_id $website_name
    
    # Check whether the website can be reached or not
    is_website_black $website_name
    
    # If the website cannot be reached, append it into websites black list
    if (( website_status > 0 )); then
        echo $website_id,$website_name,$website_hpcode,$website_return >> $WEBSITES_BLACK
    fi
done < $WEBSITES_INPUT

