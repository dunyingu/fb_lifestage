#!/bin/bash

if [ $# -lt 1 ]; then
	echo "Usage: ./run_pyscript.sh <filename> [arguments]"
	exit 1
fi

if [ ! -f $1 ]; then
	echo "cannot access ‘$1’: No such file"
	exit 1
fi

# set file references
CONNECTION_FILE='/tmp/cfg/connection-file.json'
DNS_FILE='/etc/dnsmasq.conf'
HOSTS_FILE='/etc/hosts'
SCRIPT_NAME=$1

# set container options
DOCKER_NAME='voodoo'$RANDOM

CONTROL_PORT=`jq '.control_port' < $CONNECTION_FILE`
SHELL_PORT=`jq '.shell_port' < $CONNECTION_FILE`
STDIN_PORT=`jq '.stdin_port' < $CONNECTION_FILE`
HB_PORT=`jq '.hb_port' < $CONNECTION_FILE`
IOPUB_PORT=`jq '.iopub_port' < $CONNECTION_FILE`

DNS_SERVER=`cat $DNS_FILE | grep ntp.org | cut -d '/' -f3`
EMAIL_IP=`cat $HOSTS_FILE | grep email | cut -d ' ' -f1`
IMAGE_URL=`docker image ls --format {{.Repository}}:{{.Tag}}`

# start container
docker run -w /tmp/app/ -t -d --net sagemaker-local -v $CONNECTION_FILE:/tmp/cfg/connection-file.json \
--add-host email-smtp.ap-southeast-2.amazonaws.com:$EMAIL_IP \
--name $DOCKER_NAME \
--dns "$DNS_SERVER" \
-p $CONTROL_PORT:$CONTROL_PORT \
-p $SHELL_PORT:$SHELL_PORT \
-p $STDIN_PORT:$STDIN_PORT \
-p $HB_PORT:$HB_PORT \
-p $IOPUB_PORT:$IOPUB_PORT \
$IMAGE_URL 1>/dev/null

# upload all files except certain directories
for file in *; do
	if [ $file != 'kernels' ] && [ $file != 'lost+found' ]; then
		docker cp $file $DOCKER_NAME:/tmp/app/;
	fi;
done

# execute script
docker exec -it $DOCKER_NAME python /tmp/app/$@
# docker exec -it $DOCKER_NAME python $@

# stop container
# docker stop $DOCKER_NAME -- slow
docker kill $DOCKER_NAME 1>/dev/null
docker container rm $DOCKER_NAME 1>/dev/null