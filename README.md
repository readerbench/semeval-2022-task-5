## Installing APEX
        export CUDA_HOME=/usr/local/cuda-10.2
        git clone https://github.com/NVIDIA/apex
        pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

colab:
%%writefile setup.sh

#export CUDA_HOME=/usr/local/cuda-10.0
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
!sh setup.sh

## TODO: 
* Cleanup cele cu situri si alte rahaturi in ele
* lista de objects de ignorat
* de incercat si cu sentiment

