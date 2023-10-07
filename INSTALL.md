참고 사이트: 

    https://medium.com/@soji4u2c/how-to-install-the-nvidia-cuda-driver-12-0-toolkit-cudnn-8-8-1-3-on-wsl2-in-year-2023-23165024dc16

- 위 참고 사이트를 기본적으로 그대로 따라함

- 두 가지 문제 발생

    - 첫째: 아래 명령문 실행 후 "sudo apt-get update" 실행할 때 gpg 키 오류 발생
    
            sudo sh -c 'echo "deb  http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-machine-learning.list'`

        - 해결책 참고: https://velog.io/@offsujin/Ubuntu-GPG-error-%ED%95%B4%EA%B2%B0
    
    - 둘째: libcudnn8 관련 설치할 때 8.8.1.3-1+cuda12.0 사용하면 안됨. 
    
        - 먼저 cuDNN 8.8.1 library 가 아닌 cuDNN 8.9.5 for CUDA 12.2 (for ubuntu 20.04) 다운로드해서 설치할 것.
            - cuDNN 8.8.1을 설치해도 문제되지 않았지만 그게 맞을 것 같음.
        
        - 아래 명령문 실행하여 설치가능 패키지 확인할 것
        
                apt-cache policy libcudnn8
      
        - 그런 다음 cuDNN 8.9.5와의 최신 조합 사용할 것:  8.9.5.29-1+cuda12.2
           
       
- 이제 miniconda(또는 anaconda), jupyter, tensorflow 설치
    
    - (miniconda 대신에 anaconda 를 사용하면?)

    - miniconda 와 jupyter 설치: https://github.com/codingalzi/t81_558_deep_learning/blob/master/install/tensorflow-install-march-2023.ipynb
  
            curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
    
            bash Miniconda3-latest-Linux-x86_64.sh
    
            conda install -y jupyter
  
    - tensorflow 설치: https://www.tensorflow.org/install/pip
  
            pip install --upgrade pip

            pip install tensorflow[and-cuda]

    