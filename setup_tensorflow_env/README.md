# Training the tflite models

At the end of this you should have your .tflite models saved locally. You need to have completed `generate_tf_records` before this.

## Get a machine

For reproducability purposes and my laptop being slow I have decided to launch an Ubuntu instance on EC2 for training machine learning model.

I used `Ubuntu Server 20.04 LTS (HVM), SSD Volume Type - ami-09e67e426f25ce0d7 (64-bit x86) / ami-00d1ab6b335f217cf (64-bit Arm)` instance with following options:
 - c4.4xlarge
 - 40 GB of storage
 - Security group so that you can ssh into it.
 - save the pem key

I have successfully tested on my local Ubuntu machine training Tensorflow models so an EC2 instance is not necessary. The issue I ran into with Tensorflow is it is sensitive to various environments and can be finicky to setup. Once again this EC2 setup described here is for reproducability.

## Create Tensorflow environments

We will be creating environment Tensorflow 1 and Tensorflow 2 environments to train two models.

ssh into your machine:

```
ssh -i /full/path/to/pem/key ubuntu@public_ip
```

### Install Conda

We will be using two different versions of Tensorflow for which we will want two separate environments:

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```

You will have to restart shell to use conda. Then run

```
sudo snap install protobuf --classic
sudo apt update
sudo apt install build-essential -y
sudo apt install libgl1-mesa-glx -y
```

### Tensorflow 2

Run:

```
conda create --name tensorflow_2 python=3.6 -y
conda install -n tensorflow_2 tensorflow=2.2.0 tensorflow-datasets tensorflow-estimator tensorflow-hub tensorflow-metadata -y
conda activate tensorflow_2
```

```
cd ~
mkdir -p tensorflow_2
cd tensorflow_2
git clone https://github.com/tensorflow/models.git
cd models
git checkout 594341996acb4f419dbdfbaae258e4b230cb9ea1
git submodule update --init --recursive
cd research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python3 -m pip install --upgrade pip
python3 -m pip install .
python3 object_detection/builders/model_builder_tf2_test.py
pip install opencv-python
pip install opencv-contrib-python
pip install absl-py
pip install tflite_support
pip install tensorflow==2.3.0
```

### Tensorflow 1

Run:

```
conda create --name tensorflow_1 python=3.7 -y
conda activate tensorflow_1
conda install -c conda-forge tensorflow=1.15 -y
pip install tensorflow-object-detection-api
```


```
conda deactivate
conda activate tensorflow_1
cd ~
mkdir -p tensorflow_1
cd tensorflow_1
git clone https://github.com/tensorflow/models.git
cd models
git submodule update --init --recursive
cd research
protoc object_detection/protos/*.proto --python_out=. # from within models/research
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip install numpy==1.17.4
cp object_detection/packages/tf1/setup.py .
python -m pip install . #--use-feature=2020-resolver .
pip install tf-models-official
python object_detection/builders/model_builder_tf1_test.py
pip install tensorflow==1.15.5
pip install tensorflow-object-detection-api
pip install pycocotools==2.0.2
pip install numpy==1.17.4
pip install tflite_support
```


## Copy over training and config data

If you are using your own classes or different number of classes or object types you will have to modify `num_classes` entry in `pipeline*.config`, `labelmap.txt` and `tf_label_map.pbtxt`.

Modify the `scp_script.sh`:

```
user=ubuntu
ip_address=public_ip
key_path=/full/path/to/pem
repository_directory=/path/to/this/repository/machineLearningStarter
```

Run `./scp_script.sh`


## Train Tensorflow 2 Model

First confirm your setup works. Change `~/tensorflow_2/pipeline_fpn.config` to have `total_steps` and `num_steps` to be 500 and `warmup_steps` to be `100`. This will run through a basic run through 

### Train model

```
cd ~/tensorflow_2
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
tar -xzvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

outdir=training
mkdir -p $outdir
python3 models/research/object_detection/model_main_tf2.py --alsologtostderr --model_dir=$outdir --checkpoint_every_n=500  --pipeline_config_path=/home/ubuntu/tensorflow_2/pipeline_fpn.config \ | tee /tmp/train.log
```

### Convert to TFLITE format

```
pip install tensorflow==2.4.0
cd ~/tensorflow_2
export_dir=export_tflite
mkdir -p $export_dir
python3 models/research/object_detection/export_tflite_graph_tf2.py --pipeline_config_path=/home/ubuntu/tensorflow_2/pipeline_fpn.config  --trained_checkpoint_dir=$outdir --output_directory=$export_dir
convert_dir=convert_dir
mkdir -p $convert_dir
python3 convert_to_tflite.py $export_dir/saved_model $convert_dir
result=result
mkdir -p $result
python3 script.py $convert_dir/model.tflite labelmap.txt $result/result.tflite
cp labelmap.txt $result
```

### Copy over the model

```
scp -i /full/path/to/pem/key ubuntu@public_ip:~/tensorflow_2/result/* local/directory/of/your/choice
```


## Train Tensorflow 1 Model

First confirm your setup works.

### Train model

```
cd ~/tensorflow_1
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
tar -xzvf  ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz

out_dir=~/tensorflow_1/training
python  models/research/object_detection/model_main.py --logtostderr --train_dir=$out_dir --model_dir=$out_dir --pipeline_config_path=/home/ubuntu/tensorflow_1/pipeline.config
```

### Convert to TFLITE format

```
conda activate tensorflow_1
cd ~/tensorflow_1
export_dir=export_tflite
mkdir -p $export_dir
cd models/research/object_detection
python export_tflite_ssd_graph.py  --pipeline_config_path /home/ubuntu/tensorflow_1/pipeline.config  --trained_checkpoint_prefix /home/ubuntu/tensorflow_1/training/model.ckpt-6000 --output_directory /home/ubuntu/tensorflow_1/$export_dir

cd ~/tensorflow_1
convert_dir=convert_dir
mkdir -p $convert_dir

tflite_convert --graph_def_file=$export_dir/tflite_graph.pb --output_file=$convert_dir/model.tflite --output_format=TFLITE --input_shapes=1,320,320,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=FLOAT --mean_values=128  --std_dev_values=127 --change_concat_input_ranges=false  --allow_custom_ops

result=result
mkdir -p $result
python3 script.py $convert_dir/model.tflite labelmap.txt $result/result.tflite
cp labelmap.txt $result
```

### Copy over the model

```
scp -i /full/path/to/pem/key ubuntu@public_ip:~/tensorflow_1/result/* local/directory/of/your/choice
```
