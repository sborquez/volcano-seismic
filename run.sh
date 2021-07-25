name="volcano-seismic_classifier" 
docker_image="sborquez/datascience:latest-gpu"

docker run --name $name --gpus all --rm -it\
            -p 8888:8888 -p 8787:8787 -p 8786:8786\
            -v "$(pwd):/rapids/host/notebooks"\
            $docker_image