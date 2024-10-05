# Author: Junhao Bai
# Date: 2024-10-05 21:22:26

# Use the official Ubuntu 24.04 image
FROM ubuntu:24.04

# Set the Mirrors
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    axel \
    git \
    zsh \
    htop \
    vim \
    tmux \
    tree \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Node.js and npm

# Use nvm to manage Node.js versions
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
RUN echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.zshrc
RUN echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> ~/.zshrc
RUN echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"' >> ~/.zshrc
RUN /bin/zsh -c "source ~/.zshrc && nvm install node"

# Set npmmirror
RUN npm config set registry https://registry.npmmirror.com

# Set up node-red
RUN npm install -g --unsafe-perm node-red

# Set up MongoDB and create admin user
RUN apt-get update && apt-get install -y mongodb
RUN mkdir -p /data/db
RUN mongod --fork --logpath /var/log/mongodb.log
RUN mongo admin --eval 'db.createUser \
    ({user: "admin", pwd: "admin", roles: ["userAdminAnyDatabase"]})'
RUN mongod --shutdown

# Set up Oh My Zsh
RUN git clone https://mirrors.tuna.tsinghua.edu.cn/git/ohmyzsh.git &&\
    cd ohmyzsh/tools &&\
    REMOTE=https://mirrors.tuna.tsinghua.edu.cn/git/ohmyzsh.git sh install.sh

# Set up the default shell to Zsh
RUN chsh -s /usr/bin/zsh

# mount ssh key
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    chmod 700 /root/.ssh && \
    chmod 600 /root/.ssh/*