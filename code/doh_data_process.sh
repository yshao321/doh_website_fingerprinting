#!/bin/bash
# 
# This is a script to process DNS-over-HTTPS traffic
# 

DOH_SVR_IP_1="104.16.248.249"
DOH_SVR_IP_2="104.16.249.249"

parse_line () {
	oneline=${line}
	#echo "$oneline"

	if [[ $oneline == *" > $DOH_SVR_IP_1"* || $oneline == *" > $DOH_SVR_IP_2"* ]]; then
		len=$(echo "$oneline" | sed -n 's/.*length \([0-9]*\).*/\1/p')
		if [ ! -z $len ] && [ $len -gt 0 ]; then
			if [ $site_valid -gt 0 ]; then
				site_data+=", "
			fi
			site_data+=$len
			((site_valid++))
		fi
	fi

	if [[ $oneline == *"IP $DOH_SVR_IP_1"* || $oneline == *"IP $DOH_SVR_IP_2"* ]]; then
		len=$(echo "$oneline" | sed -n 's/.*length \([0-9]*\).*/\1/p')
		if [ ! -z $len ] && [ $len -gt 0 ]; then
			if [ $site_valid -gt 0 ]; then
				site_data+=", "
			fi
			site_data+="-$len"
			((site_valid++))
		fi
	fi
}

process_stdin () {
	# If a pipe exists on stdin, it is for real-time prediction

	mkdir -p ../temp

	while true; do

		site_data="{ \n\n\"0\" : { \"lengths\" : [ "
		site_valid=0

		# Analyze real-time data when no record for 5 seconds ==> predict fast
		while read -t 5 line; do
			echo -n "." >&2
			parse_line
		done
		echo "*" >&2

		# Analyze real-time data with more than 5 TLS records ==> dicard noise
		if [ $site_valid -gt 5 ]; then
			site_data+=" ] } \n\n}"

			current_time="`date +%Y%m%d%H%M%S`"
			current_file="../temp/$current_time.json"
			echo -e "$site_data" > $current_file
			echo "$current_file"
		fi
	done
}

process_collection () {
	echo "Processing collection..."

	mkdir -p ../dataset
	mkdir -p ../collection

	cd ../collection

	while read dir_name; do

		### bypass processed directory ###
		if [ -f "../dataset/$dir_name.json" ]; then
			continue
		fi

		### one directory for one sample ###
		echo "$dir_name"
		sample_data="{ \n"
		sample_valid=0

		cd $dir_name

		while read file_name; do

			### one pcap file for one site ###
			echo "$file_name"
			site_name=${file_name/.pcap}
			#site_class=$(grep -w -n ${site_name:2} ../websites.txt | head -n 1 | cut -d: -f1)
			site_class=${site_name:2}
			site_valid=0

			if [ $sample_valid == 1 ]; then
				site_data=",\n\"$site_class\" : { \"lengths\" : [ "
			else
				site_data=" \n\"$site_class\" : { \"lengths\" : [ "
			fi

			while read line; do
				parse_line
			done < <(tcpdump -r $file_name 2>/dev/null)

			### site valid ###
			if [ $site_valid -gt 0 ]; then
				site_data+=" ] } \n"

				sample_data+=$site_data
				sample_valid=1
			fi

		done < <(find ./ -name "*.pcap" -type f)

		### sample valid ###
		if [ $sample_valid == 1 ]; then
			sample_data+="\n}"
			echo -e "$sample_data" > ../../dataset/$dir_name.json
		fi

		cd ..

	done < <(find ./ -mindepth 1 -maxdepth 1 -type d)

	cd ..
	echo "Processing done!!!"
}

# Check to see if a pipe exists on stdin.
if [ -p /dev/stdin ]; then
	process_stdin
else
	process_collection
fi

