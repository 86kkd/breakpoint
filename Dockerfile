FROM  pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

LABEL Name=breakout Version=0.0.1


# change source list
# skip timezone setting
ENV DEBIAN_FRONTEND=noninteractive 
COPY sources.list /etc/apt/sources.list
RUN apt update && apt upgrade -y --no-install-recommends

# install package
RUN apt install -y --no-install-recommends \
    curl fortune sudo cowsay git fish \
    # ffmpeg libsm6 libxext6 \
    ssh
RUN rm -rf /var/lib/apt/lists/*

# install fortunes (for fun)
RUN git clone  https://github.com/ruanyf/fortunes.git && \
    mkdir -p /usr/share/games/fortunes && \
    mv fortunes/data/* /usr/share/games/fortunes && \
    rm -rf fortunes

# create user and user group
RUN addgroup --gid 1000 docker && \ 
    adduser --uid 1000 --ingroup docker  \
    --home /home/docker --shell /bin/bash \
    --disabled-password\
    --gecos "" docker && \ 
    echo 'docker:docker' | chpasswd && \
    echo 'root:docker' | chpasswd && \
    usermod -aG sudo docker

ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'docker  ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers

# install fixuid
RUN USER=docker && \
    GROUP=docker && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.6.0/fixuid-0.6.0-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml

USER docker:docker

ENTRYPOINT ["fixuid"]

# set work dir
WORKDIR /home/docker
COPY . /home/docker

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt


CMD ["sh", "-c", "/usr/games/fortune -a | /usr/games/cowsay"]