#!/bin/bash

user=ubuntu
ip_address=
key_path=
repository_directory=

ssh -i $key_path $user@$ip_address 'mkdir -p ~/tf_records'
scp -i $key_path $repository_directory/generate_tf_records/tf_records/* $user@$ip_address:~/tf_records
ssh -i $key_path $user@$ip_address 'mkdir -p ~/tf_records_training'
scp -i $key_path $repository_directory/generate_tf_records/tf_records_training/* $user@$ip_address:~/tf_records_training

ssh -i $key_path $user@$ip_address 'mkdir -p ~/tensorflow_2'
scp -i $key_path $repository_directory/setup_tensorflow_env/tensorflow_2/* $user@$ip_address:~/tensorflow_2

ssh -i $key_path $user@$ip_address 'mkdir -p ~/tensorflow_1'
scp -i $key_path $repository_directory/setup_tensorflow_env/tensorflow_1/* $user@$ip_address:~/tensorflow_1

